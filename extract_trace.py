import sqlite3

conn = sqlite3.connect("demo.db")
cur = conn.cursor()

query = """
SELECT
    timestamp,
    action,
    overall_tier,
    top_factor,
    notes
FROM decision_trace
WHERE child_id = ?
ORDER BY timestamp
"""

cur.execute(query, ("ECD-0002",))

rows = cur.fetchall()

print("=== Transition History for ECD-0002 ===\n")

for timestamp, action, tier, factor, notes in rows:
    print(f"Timestamp : {timestamp}")
    print(f"Action    : {action}")
    print(f"Tier      : {tier}")
    print(f"Top Factor: {factor}")
    print(f"Notes     : {notes}")
    print("-" * 50)

conn.close()