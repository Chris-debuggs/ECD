from risk_engine import load_model
import pandas as pd
artifact=load_model()

print("Model Loaded")
'''
print(type(artifact))
print(artifact["models"].keys())
print(artifact["feature_map"])
'''
cog_model=artifact["models"]["cognitive"]
mor_model=artifact["models"]["motor"]
soc_model=artifact["models"]["socio_emotional"]

X = pd.DataFrame([{
    "cog_lang_milestone": 1,
    "cog_memory_recall": 1,
    "cog_problem_solving": 1,
    "cog_attention_span": 1,
    "cog_learning_adapt": 1
}])

Y= pd.DataFrame([{
    "mot_gross_motor": 1,
    "mot_fine_motor": 1,
    "mot_balance":  1,
    "mot_hand_eye": 1,
    "mot_body_aware": 1
}])

Z= pd.DataFrame([{
     "se_social_play":1,
     "se_emotion_reg": 1,
     "se_peer_interact":1,
     "se_attachment":1,
     "se_self_care":1
}])

test_X=cog_model.predict_proba(X)
test_Y=mor_model.predict_proba(Y)
test_Z=soc_model.predict_proba(Z)
print(f"The raw data for cognative model after prediction : {test_X}")
print(f"The raw data for motor model after prediction : {test_Y}")
print(f"The raw data for socio emotional model after prediction : {test_Z}")
