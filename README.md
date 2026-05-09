# Flask ML API — Eventzilla
## Setup rapide (3 étapes)

### Étape 1 — Sauvegarder les modèles depuis le notebook
Colle le contenu de `save_models.py` dans une **nouvelle cellule à la fin** de ton notebook Jupyter, et exécute-la.
Tu dois voir apparaître un dossier `models/` avec 3-5 fichiers `.pkl`.

### Étape 2 — Installer les dépendances Flask
Ouvre un terminal dans le dossier `flask_api/` et lance :
```bash
pip install -r requirements.txt
```

### Étape 3 — Lancer l'API
```bash
# Copie le dossier models/ ici aussi
cp -r ../models ./models

# Lance Flask
python app.py
```
L'API tourne sur : http://localhost:5000

---
## Tester l'API (curl)

### Health check
```bash
curl http://localhost:5000/health
```

### Prédiction Classification
```bash
curl -X POST http://localhost:5000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{
    "taux_reclamation_hist": 0.15,
    "taux_acceptation_prest": 0.85,
    "taux_annulation_hist": 0.05,
    "nb_reservations_total": 120,
    "qualite_score": 0.72,
    "month": 6,
    "quarter": 2,
    "event_type_enc": 1,
    "service_type_enc": 2
  }'
```

### Prédiction Régression
```bash
curl -X POST http://localhost:5000/predict/regression \
  -H "Content-Type: application/json" \
  -d '{
    "taux_reclamation_hist": 0.20,
    "taux_acceptation_prest": 0.75,
    "taux_annulation_hist": 0.10,
    "nb_reservations_total": 80,
    "qualite_score": 0.60,
    "bookedcount": 5,
    "month": 3,
    "quarter": 1,
    "event_type_enc": 0,
    "service_type_enc": 1
  }'
```

### Prédiction Clustering
```bash
curl -X POST http://localhost:5000/predict/clustering \
  -H "Content-Type: application/json" \
  -d '{
    "taux_reclamation_hist": 0.05,
    "taux_acceptation_prest": 0.90,
    "taux_annulation_hist": 0.03,
    "nb_reservations_total": 200,
    "qualite_score": 0.88
  }'
```

### Les 3 prédictions en une fois (pour n8n)
```bash
curl -X POST http://localhost:5000/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "taux_reclamation_hist": 0.10,
    "taux_acceptation_prest": 0.80,
    "taux_annulation_hist": 0.05,
    "nb_reservations_total": 150,
    "qualite_score": 0.75,
    "bookedcount": 8,
    "month": 4,
    "quarter": 2,
    "event_type_enc": 1,
    "service_type_enc": 0
  }'
```
