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

# ==================== INITIALISATION ====================
app = FastAPI(title="Drug Classification API")

# Configurer les templates HTML
templates = Jinja2Templates(directory="templates")

# ==================== CHARGEMENT DU MODÈLE ====================
MODEL_PATH = "models/logistic_regression_model.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("=" * 50)
    print("✅ MODÈLE CHARGÉ AVEC SUCCÈS")
    print(f"   Fichier: {MODEL_PATH}")
    if hasattr(model, 'classes_'):
        print(f"   Classes: {model.classes_}")
    print("=" * 50)
except Exception as e:
    print(f"❌ ERREUR: Impossible de charger le modèle: {e}")
    model = None

# ==================== FONCTIONS UTILES ====================
def encode_features(age, sex, bp, cholesterol, na_to_k):
    """Convertit les données texte en nombres pour le modèle"""
    sex_code = 1 if sex == "M" else 0
    
    bp_codes = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
    bp_code = bp_codes.get(bp, 1)
    
    chol_codes = {"NORMAL": 0, "HIGH": 1}
    chol_code = chol_codes.get(cholesterol, 0)
    
    return np.array([[age, sex_code, bp_code, chol_code, na_to_k]])

# ==================== ROUTES PRINCIPALES ====================
@app.get("/", response_class=HTMLResponse) 
def page_accueil(request: Request):
    """Page d'accueil avec formulaire"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "titre": "Prédiction de Médicaments",
            "modele_charge": model is not None
        }
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
    """Reçoit les données du formulaire et fait une prédiction"""
    
    # Vérifier si le modèle est disponible
    if model is None:
        return templates.TemplateResponse(
            "erreur.html",
            {
                "request": request,
                "titre": "Erreur",
                "message": "Le modèle de prédiction n'est pas disponible."
            }
        )
    
    try:
        # Préparer les données pour l'affichage
        info_patient = {
            "Âge": f"{Age} ans",
            "Sexe": "Homme" if Sex == "M" else "Femme",
            "Pression artérielle": BP,
            "Cholestérol": Cholesterol,
            "Ratio Na/K": f"{Na_to_K:.2f}"
        }
        
        # Convertir les données pour le modèle
        donnees_codees = encode_features(Age, Sex, BP, Cholesterol, Na_to_K)
        
        # Faire la prédiction
        prediction = model.predict(donnees_codees)[0]
        
        # Calculer les probabilités si disponible
        probabilites = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(donnees_codees)[0]
            for classe, prob in zip(model.classes_, probs):
                probabilites[str(classe)] = f"{prob*100:.1f}%"
        
        # Afficher le résultat
        return templates.TemplateResponse(
            "resultat.html",
            {
                "request": request,
                "titre": "Résultat de la Prédiction",
                "prediction": prediction,
                "patient": info_patient,
                "probabilites": probabilites,
                "confidence": max(probs)*100 if 'probs' in locals() else 100
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "erreur.html",
            {
                "request": request,
                "titre": "Erreur de traitement",
                "message": f"Une erreur est survenue: {str(e)}"
            }
        )

@app.get("/api/predict")
def api_predict(Age: int, Sex: str, BP: str, Cholesterol: str, Na_to_K: float):
    """Version API pour les applications"""
    if model is None:
        return {"erreur": "Modèle non disponible"}
    
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
    """Vérifie que l'API fonctionne"""
    return {
        "status": "OK",
        "modele_charge": model is not None,
        "message": "API Drug Classification opérationnelle"
    }

@app.get("/docs")
def documentation():
    """Redirige vers la documentation Swagger"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 LANCEMENT DE L'API DRUG CLASSIFICATION")
    print("="*50)
    print("📝 Documentation: http://localhost:8000/docs")
    print("🌐 Interface web: http://localhost:8000")
    print("🩺 Vérification: http://localhost:8000/health")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
