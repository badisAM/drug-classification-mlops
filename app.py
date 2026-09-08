"""
app.py - API FastAPI pour la prédiction de médicaments
Atelier 4 - Exposition des fonctions via FastAPI
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI(title="Drug Classification API")
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/logistic_regression_model.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("=" * 50)
    print("MODELE CHARGE AVEC SUCCES")
    print(f"   Fichier: {MODEL_PATH}")
    if hasattr(model, 'classes_'):
        print(f"   Classes: {model.classes_}")
    print("=" * 50)
except Exception as e:
    print(f"ERREUR: Impossible de charger le modele: {e}")
    model = None

def encode_features(age, sex, bp, cholesterol, na_to_k):
    sex_code = 1 if sex == "M" else 0
    bp_codes = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
    bp_code = bp_codes.get(bp, 1)
    chol_codes = {"NORMAL": 0, "HIGH": 1}
    chol_code = chol_codes.get(cholesterol, 0)
    return np.array([[age, sex_code, bp_code, chol_code, na_to_k]])

@app.get("/", response_class=HTMLResponse)
def page_accueil(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"titre": "Prediction de Medicaments", "modele_charge": model is not None}
    )

@app.post("/predict", response_class=HTMLResponse)
def faire_prediction(
    request: Request,
    Age: int = Form(...),
    Sex: str = Form(...),
    BP: str = Form(...),
    Cholesterol: str = Form(...),
    Na_to_K: float = Form(...)
):
    if model is None:
        return templates.TemplateResponse(
            request, "erreur.html",
            {"titre": "Erreur", "message": "Le modele de prediction n'est pas disponible."}
        )
    try:
        info_patient = {
            "Age": f"{Age} ans",
            "Sexe": "Homme" if Sex == "M" else "Femme",
            "Pression arterielle": BP,
            "Cholesterol": Cholesterol,
            "Ratio Na/K": f"{Na_to_K:.2f}"
        }
        donnees_codees = encode_features(Age, Sex, BP, Cholesterol, Na_to_K)
        prediction = model.predict(donnees_codees)[0]
        probabilites = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(donnees_codees)[0]
            for classe, prob in zip(model.classes_, probs):
                probabilites[str(classe)] = f"{prob*100:.1f}%"
        return templates.TemplateResponse(
            request, "resultat.html",
            {
                "titre": "Resultat de la Prediction",
                "prediction": prediction,
                "patient": info_patient,
                "probabilites": probabilites,
                "confidence": max(probs)*100 if 'probs' in locals() else 100
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request, "erreur.html",
            {"titre": "Erreur de traitement", "message": f"Une erreur est survenue: {str(e)}"}
        )

@app.get("/api/predict")
def api_predict(Age: int, Sex: str, BP: str, Cholesterol: str, Na_to_K: float):
    if model is None:
        return {"erreur": "Modele non disponible"}
    try:
        donnees = encode_features(Age, Sex, BP, Cholesterol, Na_to_K)
        prediction = model.predict(donnees)[0]
        return {
            "prediction": prediction,
            "donnees": {"Age": Age, "Sex": Sex, "BP": BP, "Cholesterol": Cholesterol, "Na_to_K": Na_to_K},
            "status": "success"
        }
    except Exception as e:
        return {"erreur": str(e)}

@app.get("/health")
def sante_api():
    return {
        "status": "OK",
        "modele_charge": model is not None,
        "message": "API Drug Classification operationnelle"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
