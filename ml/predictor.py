import os, pickle
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "disease_model.pkl")

def model_status():
    return {"trained_model_available": os.path.exists(MODEL_PATH),
            "mode": "trained-demo-model" if os.path.exists(MODEL_PATH) else "safe-fallback",
            "warning": "Synthetic demo data only; not clinically validated."}

def predict_category(analysis):
    if not analysis.get("symptoms") or analysis.get("urgent_flag") or analysis.get("breathing_difficulty"):
        return {"category":"Unable to classify safely","confidence":None,"status":"needs_professional_review",
                "explanation":"Insufficient or potentially urgent information. Seek professional assessment."}
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH,"rb") as f: model = pickle.load(f)
        text = " ".join(analysis["symptoms"])
        pred = model.predict([text])[0]
        confidence = round(float(max(model.predict_proba([text])[0])),3) if hasattr(model,"predict_proba") else None
        return {"category":pred,"confidence":confidence,"status":"demo_prediction",
                "explanation":"Synthetic demonstration model output only; not a diagnosis."}
    return {"category":"Unclassified – train model first","confidence":None,"status":"model_not_trained",
            "explanation":"Run python train_model.py to create the demonstration model."}
