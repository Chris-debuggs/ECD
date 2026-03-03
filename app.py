"""
app.py — ECD Decision Support Layer: Streamlit MVP Dashboard

Run: streamlit run app.py
Prerequisite: python seed_demo_db.py (run once to populate demo.db)

Views:
  AWW Role:       New Screening | My Children
  Supervisor Role: Block Dashboard | Child Trajectories | High-Risk Alerts
  All Roles:      Audit Trail | About This System
"""

import os
import uuid
import json
import sqlite3
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECD Decision Support Layer",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from our modules ────────────────────────────────────────────────
from schema import (
    get_connection, DB_PATH, TIER_LABELS, TIER_COLOURS,
    FEATURE_LABELS, COGNITIVE_FEATURES, MOTOR_FEATURES, SOCIO_FEATURES
)
from decision_trace import (
    log_screening, log_consent, log_referral, get_trace_as_display_df
)
from cohort_aggregator import (
    block_summary, district_rollup, child_list_for_supervisor
)
from trajectory_engine import (
    trajectory_summary, impact_metrics
)


# ── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading risk model...")
def load_artifact():
    from risk_engine import load_model
    return load_model()


# ── Load DB data (cached with TTL) ───────────────────────────────────────────
@st.cache_data(ttl=30)
def load_data():
    conn = get_connection(DB_PATH)
    children_df    = pd.read_sql("SELECT * FROM children", conn)
    assessments_df = pd.read_sql("SELECT * FROM assessments", conn)
    risk_df        = pd.read_sql("SELECT * FROM risk_results", conn)
    traj_df        = pd.read_sql("SELECT * FROM developmental_trajectories", conn)
    consents_df    = pd.read_sql("SELECT * FROM consents", conn)
    conn.close()
    return children_df, assessments_df, risk_df, traj_df, consents_df


def refresh_data():
    load_data.clear()


# ── Helpers ──────────────────────────────────────────────────────────────────
TIER_EMOJI = {0: "🟢", 1: "🟡", 2: "🔴", 3: "🟣"}
DOMAIN_NICE = {"cognitive": "Cognitive", "motor": "Motor", "socio_emotional": "Socio-Emotional"}

def tier_badge(tier: int) -> str:
    colour = TIER_COLOURS.get(tier, "#999")
    label  = TIER_LABELS.get(tier, "Unknown")
    return f'<span style="background:{colour};color:white;padding:4px 12px;border-radius:12px;font-weight:bold;">{label}</span>'

def disclaimer():
    st.warning(
        "⚠️ **Screening notice:** This is a developmental screening flag, not a clinical diagnosis. "
        "All High and Critical flags require supervisor review and referral to a qualified health professional.",
        icon=None
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=60)
    st.markdown("### ECD Decision Support Layer")
    st.markdown("*ICDS / WD&CW Innovation Challenge*")
    st.divider()

    role = st.selectbox("👤 Select Role", ["AWW (Field Worker)", "Supervisor"], key="role_selector")
    is_aww = role.startswith("AWW")

    offline_mode = st.toggle("📴 Offline Mode", value=False,
                             help="Simulates device with no connectivity. Provisional risk only.")
    if offline_mode:
        st.warning("Offline Mode — Provisional Risk Only. Sync when connected.")

    st.divider()

    children_df, assessments_df, risk_df, traj_df, consents_df = load_data()
    all_blocks = sorted(children_df["block_id"].unique().tolist())

    if not is_aww:
        selected_block = st.selectbox("📍 Filter by Block", ["All Blocks"] + all_blocks)
    else:
        my_anganwadis = sorted(children_df["anganwadi_id"].unique().tolist())
        selected_awc  = st.selectbox("🏘️ My Anganwadi", my_anganwadis)

    st.divider()
    st.caption(f"DB: {os.path.basename(DB_PATH)}  |  Children: {len(children_df)}")


# ── Tab layout ────────────────────────────────────────────────────────────────
if is_aww:
    tabs = st.tabs(["📋 New Screening", "👶 My Children", "📜 Audit Trail", "ℹ️ About"])
    tab_screen, tab_mychildren, tab_audit, tab_about = tabs
else:
    tabs = st.tabs(["📊 Block Dashboard", "📈 Child Trajectories",
                    "🚨 High-Risk Alerts", "📜 Audit Trail", "ℹ️ About"])
    tab_dash, tab_traj, tab_alerts, tab_audit, tab_about = tabs


# ═══════════════════════════════════════════════════════════════════════════
# AWW: NEW SCREENING
# ═══════════════════════════════════════════════════════════════════════════
if is_aww:
    with tab_screen:
        st.header("📋 New Child Screening")
        disclaimer()

        # -- Step 1: Consent ---------------------------------------------------
        with st.expander("Step 1: Consent Capture", expanded=True):
            awc_children = children_df[children_df["anganwadi_id"] == selected_awc]
            existing_consents = consents_df["child_id"].tolist()

            child_options = awc_children[["child_id", "name", "age_months"]].copy()
            child_options["label"] = child_options.apply(
                lambda r: f"{r['name']} ({r['age_months']} months)", axis=1
            )
            child_label_map = dict(zip(child_options["label"], child_options["child_id"]))

            sel_label = st.selectbox("Select Child", ["-- Select --"] + child_options["label"].tolist())
            selected_child_id = child_label_map.get(sel_label)

            if selected_child_id:
                if selected_child_id not in existing_consents:
                    st.info("No consent on file. Please capture consent before screening.")
                    with st.form("consent_form"):
                        guardian_name   = st.text_input("Guardian Name *")
                        consent_method  = st.selectbox("Consent Method",
                                                        ["VERBAL_WITNESSED", "THUMBPRINT", "WRITTEN"])
                        consent_submit  = st.form_submit_button("✅ Capture Consent")
                        if consent_submit and guardian_name:
                            conn = get_connection(DB_PATH)
                            conn.execute(
                                """INSERT INTO consents
                                   (consent_id, child_id, guardian_name, consent_date,
                                    consent_method, aww_witness, created_at)
                                   VALUES (?,?,?,?,?,?,?)""",
                                (str(uuid.uuid4()), selected_child_id, guardian_name,
                                 date.today().isoformat(), consent_method, role,
                                 datetime.now().isoformat())
                            )
                            conn.commit()
                            conn.close()
                            log_consent("AWW", selected_child_id, guardian_name)
                            refresh_data()
                            st.success("Consent captured. You may now screen this child.")
                            st.rerun()
                else:
                    st.success("✅ Consent on file. Proceed to screening.")

        # -- Step 2: Indicators -----------------------------------------------
        if selected_child_id and selected_child_id in consents_df["child_id"].tolist():
            child_row = child_options[child_options["child_id"] == selected_child_id].iloc[0]
            age_months = int(child_row["age_months"])

            with st.form("screening_form"):
                st.subheader(f"Screening Indicators — {child_row['label']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Cognitive Domain**")
                    cog_lang    = st.selectbox("Language milestone achieved?", [1, 0],
                                               format_func=lambda x: "Yes" if x == 0 else "Not yet")
                    cog_mem     = st.slider("Memory recall score", 1, 4, 3,
                                           help="1=Poor, 4=Age-appropriate")
                    cog_prob    = st.slider("Problem solving score", 1, 4, 3)
                    cog_att     = st.selectbox("Attention task completed?", [0, 1],
                                               format_func=lambda x: "Yes" if x == 0 else "Incomplete")
                    cog_learn   = st.selectbox("Adapts to new task?", [0, 1],
                                               format_func=lambda x: "Yes" if x == 0 else "Struggles")

                with col2:
                    st.markdown("**Motor Domain**")
                    mot_gross   = st.selectbox("Gross motor milestone?", [0, 1],
                                               format_func=lambda x: "Met" if x == 0 else "Not met")
                    mot_fine    = st.slider("Fine motor skill", 1, 4, 3)
                    mot_bal     = st.selectbox("Balance task?", [0, 1],
                                               format_func=lambda x: "Met" if x == 0 else "Not met")
                    mot_hand    = st.slider("Hand-eye coordination", 1, 4, 3)
                    mot_body    = st.selectbox("Body awareness?", [0, 1],
                                               format_func=lambda x: "Met" if x == 0 else "Not met")

                with col3:
                    st.markdown("**Socio-Emotional Domain**")
                    se_play     = st.selectbox("Social play (age-appropriate)?", [0, 1],
                                               format_func=lambda x: "Yes" if x == 0 else "Limited")
                    se_emot     = st.slider("Emotion regulation", 1, 4, 3)
                    se_peer     = st.slider("Peer interaction quality", 1, 4, 3)
                    se_attach   = st.selectbox("Secure attachment observed?", [0, 1],
                                               format_func=lambda x: "Yes" if x == 0 else "Concern")
                    se_self     = st.selectbox("Self-care skills (age-appropriate)?", [0, 1],
                                               format_func=lambda x: "Met" if x == 0 else "Not met")

                submit_screening = st.form_submit_button("▶ Run Risk Assessment")

            if submit_screening:
                assessment_row = {
                    "cog_lang_milestone": cog_lang, "cog_memory_recall": cog_mem,
                    "cog_problem_solving": cog_prob, "cog_attention_span": cog_att,
                    "cog_learning_adapt": cog_learn,
                    "mot_gross_motor": mot_gross, "mot_fine_motor": mot_fine,
                    "mot_balance": mot_bal, "mot_hand_eye": mot_hand,
                    "mot_body_aware": mot_body,
                    "se_social_play": se_play, "se_emotion_reg": se_emot,
                    "se_peer_interact": se_peer, "se_attachment": se_attach,
                    "se_self_care": se_self,
                }

                with st.spinner("Running risk assessment..."):
                    if offline_mode:
                        from risk_engine import offline_provisional_predict
                        result = offline_provisional_predict(assessment_row)
                    else:
                        artifact = load_artifact()
                        from risk_engine import predict as do_predict
                        result = do_predict(artifact, assessment_row)

                # Save to DB
                assessment_id = str(uuid.uuid4())
                result_id     = str(uuid.uuid4())
                conn = get_connection(DB_PATH)
                conn.execute(
                    """INSERT INTO assessments
                       (assessment_id, child_id, assessment_date, age_at_assessment_months,
                        cog_lang_milestone, cog_memory_recall, cog_problem_solving,
                        cog_attention_span, cog_learning_adapt,
                        mot_gross_motor, mot_fine_motor, mot_balance, mot_hand_eye, mot_body_aware,
                        se_social_play, se_emotion_reg, se_peer_interact, se_attachment, se_self_care,
                        is_offline, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (assessment_id, selected_child_id, date.today().isoformat(), age_months,
                     cog_lang, cog_mem, cog_prob, cog_att, cog_learn,
                     mot_gross, mot_fine, mot_bal, mot_hand, mot_body,
                     se_play, se_emot, se_peer, se_attach, se_self,
                     int(offline_mode), datetime.now().isoformat())
                )
                conn.execute(
                    """INSERT INTO risk_results
                       (result_id, assessment_id, child_id, model_version,
                        cognitive_tier, motor_tier, socio_tier, overall_tier,
                        cognitive_prob, motor_prob, socio_prob, overall_prob,
                        shap_values_json, top_factors_json, is_provisional, computed_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (result_id, assessment_id, selected_child_id, result["model_version"],
                     result["cognitive_tier"], result["motor_tier"],
                     result["socio_tier"], result["overall_tier"],
                     float(result.get("cognitive_prob") or 0),
                     float(result.get("motor_prob") or 0),
                     float(result.get("socio_prob") or 0),
                     float(result.get("overall_prob") or 0),
                     result.get("shap_values_json", "{}"),
                     result.get("top_factors_json", "[]"),
                     int(result["is_provisional"]), datetime.now().isoformat())
                )
                conn.commit()
                conn.close()
                log_screening("AWW", selected_child_id, assessment_id,
                              result["overall_tier"], result.get("top_factors", []),
                              result["is_provisional"])
                refresh_data()

                # -- Display result -------------------------------------------
                st.divider()
                overall = result["overall_tier"]

                if offline_mode:
                    st.info("📴 **Provisional result** — computed on device. Full analysis after sync.")

                st.markdown(f"### Overall Risk: {tier_badge(overall)}", unsafe_allow_html=True)

                col_d1, col_d2, col_d3 = st.columns(3)
                for col, domain, tier_key, prob_key in [
                    (col_d1, "Cognitive",       "cognitive_tier", "cognitive_prob"),
                    (col_d2, "Motor",            "motor_tier",     "motor_prob"),
                    (col_d3, "Socio-Emotional",  "socio_tier",     "socio_prob"),
                ]:
                    t = result[tier_key]
                    p = result.get(prob_key)
                    with col:
                        st.metric(
                            label=domain,
                            value=f"{TIER_EMOJI[t]} {TIER_LABELS[t]}",
                            delta=f"Prob: {p:.2f}" if p is not None else "Offline"
                        )

                # SHAP factors
                top_factors = result.get("top_factors", [])
                if top_factors:
                    st.markdown("#### Key Risk Factors (Top 3)")
                    for i, f in enumerate(top_factors[:3], 1):
                        arrow = "↑ Increases risk" if "increases" in f["direction"] else "↓ Reduces risk"
                        st.markdown(f"**{i}.** {f['label']} — *{arrow}*")
                else:
                    st.info("Detailed explanation available after sync. (Offline mode)")

                disclaimer()

                # Referral gate for High/Critical
                if overall >= 2:
                    st.error(f"🚨 **{TIER_LABELS[overall]} risk flagged.** Action required before dismissing.")
                    action = st.radio("Select Action",
                                     ["HOME_VISIT", "PHC_REFERRAL", "NRC_REFERRAL"],
                                     format_func=lambda x: x.replace("_", " ").title())
                    if st.button("✅ Confirm Action & Save Referral"):
                        conn = get_connection(DB_PATH)
                        conn.execute(
                            """INSERT INTO referrals
                               (referral_id, child_id, assessment_id, created_at, action_type, outcome_code)
                               VALUES (?,?,?,?,?,?)""",
                            (str(uuid.uuid4()), selected_child_id, assessment_id,
                             datetime.now().isoformat(), action, 0)
                        )
                        conn.commit()
                        conn.close()
                        log_referral("AWW", selected_child_id, assessment_id, action, overall)
                        st.success(f"Referral ({action.replace('_', ' ').title()}) saved and logged.")


# ═══════════════════════════════════════════════════════════════════════════
# AWW: MY CHILDREN
# ═══════════════════════════════════════════════════════════════════════════
    with tab_mychildren:
        st.header(f"👶 My Children — {selected_awc}")
        awc_kids = children_df[children_df["anganwadi_id"] == selected_awc]

        latest_risk = (
            risk_df.sort_values("computed_at", ascending=False)
            .drop_duplicates("child_id")[["child_id", "overall_tier"]]
        ) if not risk_df.empty else pd.DataFrame(columns=["child_id", "overall_tier"])

        display = awc_kids.merge(latest_risk, on="child_id", how="left")
        display["Risk"]  = display["overall_tier"].map(
            lambda t: f"{TIER_EMOJI.get(int(t), '⬜')} {TIER_LABELS.get(int(t), 'Unknown')}"
            if pd.notna(t) else "⬜ Not Screened"
        )
        display["Age"]   = display["age_months"].astype(str) + " mo"
        display["Block"] = display["block_id"]

        cols_show = ["name", "Age", "gender", "Block", "Risk"]
        st.dataframe(
            display[cols_show].rename(columns={"name": "Name", "gender": "Gender"}),
            use_container_width=True, hide_index=True
        )
        st.caption(f"{len(display)} children in this anganwadi")


# ═══════════════════════════════════════════════════════════════════════════
# SUPERVISOR: BLOCK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
else:  # Supervisor
    with tab_dash:
        st.header("📊 Block Dashboard")

        traj_sum = trajectory_summary(traj_df) if not traj_df.empty else pd.DataFrame()
        block_df = block_summary(children_df, risk_df, assessments_df, traj_sum)
        district = district_rollup(block_df)

        # District-level KPIs
        if district:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Children Registered", district["total_registered"])
            k2.metric("Screened", district["total_screened"],
                      delta=f"{district['coverage_pct']}% coverage")
            k3.metric("High+Critical", district["n_high_critical"],
                      delta=f"{district['pct_high_critical']}% of screened",
                      delta_color="inverse")
            k4.metric("Deteriorating Trajectories", district["n_deteriorating"],
                      delta_color="inverse")

        st.divider()

        # Tier distribution pie
        if district:
            col_pie, col_table = st.columns([1, 2])
            with col_pie:
                labels = ["Low", "Medium", "High", "Critical"]
                values = [district.get(f"n_{l.lower()}", 0) for l in labels]
                colours = [TIER_COLOURS[i] for i in range(4)]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    marker_colors=colours, hole=0.4,
                    textinfo="label+percent"
                ))
                fig_pie.update_layout(title="Risk Tier Distribution",
                                      margin=dict(t=40, b=0, l=0, r=0), height=280)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_table:
                st.markdown("**Block-Level Summary**")
                if not block_df.empty:
                    disp_cols = ["block_id", "n_registered", "n_screened",
                                 "coverage_pct", "n_high_critical", "pct_high_critical",
                                 "n_deteriorating"]
                    st.dataframe(
                        block_df[disp_cols].rename(columns={
                            "block_id": "Block", "n_registered": "Registered",
                            "n_screened": "Screened", "coverage_pct": "Coverage %",
                            "n_high_critical": "High+Critical", "pct_high_critical": "% High+Critical",
                            "n_deteriorating": "Deteriorating"
                        }),
                        use_container_width=True, hide_index=True
                    )


# ═══════════════════════════════════════════════════════════════════════════
# SUPERVISOR: CHILD TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════
    with tab_traj:
        st.header("📈 Child Trajectories")

        filter_block = None if selected_block == "All Blocks" else selected_block
        traj_sum = trajectory_summary(traj_df) if not traj_df.empty else pd.DataFrame()
        child_list = child_list_for_supervisor(children_df, risk_df, traj_sum, filter_block)

        if child_list.empty:
            st.info("No children found for this selection.")
        else:
            # Child selector
            child_list["display"] = child_list.apply(
                lambda r: f"{r['name']} ({r['age_months']} mo) — {r.get('tier_label','?')} [{r['block_id']}]",
                axis=1
            )
            sel = st.selectbox("Select Child to View Trajectory", child_list["display"].tolist())
            sel_child_id = child_list[child_list["display"] == sel].iloc[0]["child_id"]

            child_traj = traj_df[traj_df["child_id"] == sel_child_id].sort_values("time_point")

            if len(child_traj) < 1:
                st.warning("No trajectory data for this child yet.")
            elif len(child_traj) == 1:
                st.info("Only one screening recorded. Two screenings required for trajectory.")
                row = child_traj.iloc[0]
                st.metric("Overall Developmental Coefficient",
                          f"{row['overall_coeff']:.3f} SD",
                          help="Z-score vs age-cohort mean. Negative = better than average.")
            else:
                # Trajectory chart
                fig = go.Figure()
                domains_plot = [
                    ("cognitive_coeff", "Cognitive", "#3498DB"),
                    ("motor_coeff",     "Motor",     "#2ECC71"),
                    ("socio_coeff",     "Socio-Emotional", "#E67E22"),
                    ("overall_coeff",   "Overall",   "#8E44AD"),
                ]
                for col, label, colour in domains_plot:
                    fig.add_trace(go.Scatter(
                        x=child_traj["time_point"],
                        y=child_traj[col],
                        name=label, line=dict(color=colour, width=2),
                        mode="lines+markers", marker=dict(size=8)
                    ))

                # Annotate risk transitions
                for _, row in child_traj.iterrows():
                    if pd.notna(row.get("risk_transition")) and row["risk_transition"]:
                        fig.add_annotation(
                            x=row["time_point"], y=float(row["overall_coeff"]),
                            text=f"⚠️ {row['risk_transition']}",
                            showarrow=True, arrowhead=2, arrowcolor="#E74C3C",
                            font=dict(color="#E74C3C", size=11),
                            bgcolor="white", bordercolor="#E74C3C", borderwidth=1,
                            ax=40, ay=-40
                        )

                fig.add_hline(y=0, line_dash="dash", line_color="grey",
                              annotation_text="Cohort average", annotation_position="top right")
                fig.update_layout(
                    title=f"Developmental Trajectory: {child_list[child_list['child_id']==sel_child_id].iloc[0]['name']}",
                    xaxis_title="Screening Date",
                    yaxis_title="Developmental Coefficient (SD from cohort mean)",
                    legend=dict(orientation="h", y=-0.2),
                    height=420, margin=dict(t=50, b=60)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics below chart
                latest = child_traj.sort_values("time_point").iloc[-1]
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Current Risk Tier", TIER_LABELS.get(int(latest["overall_tier"]), "?"))
                mc2.metric("Overall Coefficient", f"{latest['overall_coeff']:.3f} SD",
                           help="Negative = below average risk. Positive = above average risk.")
                mc3.metric("Last Transition",
                           latest.get("risk_transition") or "None detected")

            disclaimer()


# ═══════════════════════════════════════════════════════════════════════════
# SUPERVISOR: HIGH-RISK ALERTS
# ═══════════════════════════════════════════════════════════════════════════
    with tab_alerts:
        st.header("🚨 High-Risk Alerts")
        traj_sum = trajectory_summary(traj_df) if not traj_df.empty else pd.DataFrame()
        child_list = child_list_for_supervisor(children_df, risk_df, traj_sum)
        high_risk  = child_list[child_list["overall_tier"].fillna(-1) >= 2]

        if high_risk.empty:
            st.success("No High or Critical-risk children currently flagged.")
        else:
            st.error(f"⚠️ {len(high_risk)} children require review.")
            for _, row in high_risk.iterrows():
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 2, 2])
                    col_a.markdown(f"**{row['name']}** · {row['age_months']} mo · {row['block_id']}")
                    col_b.markdown(tier_badge(int(row["overall_tier"])), unsafe_allow_html=True)
                    col_c.markdown(f"Trend: _{row.get('trajectory_direction', '—')}_")
                    st.divider()

        disclaimer()


# ═══════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL (both roles)
# ═══════════════════════════════════════════════════════════════════════════
audit_tab = tab_audit

with audit_tab:
    st.header("📜 Decision Audit Trail")
    st.caption("All screening, consent, and referral actions are logged. Read-only.")

    trace_df = get_trace_as_display_df(DB_PATH)

    if trace_df.empty:
        st.info("No audit entries yet.")
    else:
        st.dataframe(trace_df, use_container_width=True, hide_index=True)
        csv = trace_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export CSV", csv, "audit_trail.csv", "text/csv")

    st.divider()
    st.markdown("""
    **Audit log design:**  
    Every screening, consent, referral, and override is logged with timestamp, role, 
    child ID, and top risk factor. In production, entries are cryptographically chained.
    See `ecd_system_architecture.md` § A.5 `audit_log` table for the full specification.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# ABOUT THIS SYSTEM (both roles)
# ═══════════════════════════════════════════════════════════════════════════
about_tab = tab_about

with about_tab:
    st.header("ℹ️ About This System")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
### What This System Does
- Screens children aged 0–72 months for **developmental risk** across three domains:
  Cognitive, Motor, and Socio-Emotional
- Produces a **screening-level risk tier**: Low / Medium / High / Critical
- Provides **explainable AI**: every prediction shows the top 3 contributing factors (SHAP)
- Tracks **longitudinal developmental trajectories** across screening visits
- Generates **block and district-level cohort analytics** for programme managers

### What This System Does NOT Do
- **It does not produce clinical diagnoses**
- It does not replace qualified health personnel
- It does not take autonomous action — every High/Critical flag requires human review

---
### Data Governance
- **Government data ownership:** All data resides on government infrastructure.
  No vendor has access to child data.
- **Consent-first:** Screening cannot begin without consent capture on file.
- **DPDP Act 2023 aligned:** Data minimisation, purpose limitation, consent lifecycle,
  and breach notification plan — see `dpdp_compliance_onepager.pdf`
- **Audit trail:** Every system action is logged with role, timestamp, and child ID.
        """)

    with col_r:
        st.markdown("### Model Information")

        try:
            artifact = load_artifact()
            metrics = artifact.get("metrics", {})
            st.markdown(f"**Model version:** `{artifact.get('model_version', 'N/A')}`")
            st.markdown(f"**Trained:** {artifact.get('trained_at', 'N/A')[:10]}")
            st.markdown(f"**Training data:** Validated synthetic ECD dataset (WHO/DASII norms)")
            st.markdown(f"**Explainability:** SHAP TreeExplainer (XGBoost)")
            st.markdown(f"**Calibration:** Isotonic regression (3-fold CV)")
            st.divider()

            if metrics:
                st.markdown("**Validation Metrics (held-out 20% split)**")
                rows_m = []
                for domain, m in metrics.items():
                    rows_m.append({
                        "Domain":          domain.replace("_", " ").title(),
                        "AUC-ROC":         m.get("auc_roc_macro", "—"),
                        "Balanced Acc.":   m.get("balanced_accuracy", "—"),
                        "FNR (Critical)":  m.get("fnr_critical", "—"),
                    })
                st.dataframe(pd.DataFrame(rows_m), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Model not loaded yet. Run seed_demo_db.py first. ({e})")

        st.markdown("""
---
### Thresholds & Sensitivity Policy
High and Critical tiers use **sensitivity-first thresholds** — the model is tuned to 
maximise recall for high-risk children, accepting a higher false-positive rate to 
minimise the risk of missing a child who needs intervention.

A **12% FNR alert threshold** for Critical tier triggers escalation to the ML team.

---
*This prototype uses validated synthetic data. Model will be recalibrated 
using real-world pilot data under a documented 4-week Shadow Deployment protocol 
before influencing live decisions.*
        """)
