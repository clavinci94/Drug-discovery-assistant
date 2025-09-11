# Drug Discovery Assistant

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

Ein End-to-End KI-gestütztes System für **Drug Discovery**.  
Die Applikation kombiniert moderne Methoden wie **Molekül-Featurization, Protein Language Models (ESM-2), Cross-Attention** und **Explainability**, um:  
- Molekül-Eigenschaften vorherzusagen  
- Protein–Ligand-Interaktionen zu modellieren  
- Neue Kandidaten-Moleküle zu generieren  
- Ergebnisse verständlich zu visualisieren  

---

## 🚀 Features
- **Molecular Property Prediction**: Vorhersage von ADMET-relevanten Eigenschaften  
- **Target–Drug Interaction (TDI)**: Cross-Attention Modell für Protein-Ligand-Bindung  
- **Explainability**: Residue-Importance Reports, UniProt-Mapping  
- **Optimization**: BRICS-basierte Molekül-Generierung + Multi-Objective Scoring  
- **User Interface**: Gradio Web-App mit Prediction, Batch, Optimization und Export  
- **Deployment-ready**: Dockerfile + CI/CD Pipeline  
- **Reports**: CSV, SDF, PNG und ZIP für Dokumentation  

---

## 📸 Screenshots

### Molekülvorhersage
![Prediction Screenshot](docs/screenshots/predict.png)

### Explainability Report
![Explainability Screenshot](docs/screenshots/explain.png)

---

## 📂 Repository Struktur
```text
├── data/                 # Rohdaten, verarbeitet & Embeddings
├── src/                  # Quellcode (Features, Modelle, Utils)
├── scripts/              # Trainings-, Optimierungs- und UI-Skripte
├── reports/              # Generierte Reports (CSV, PNG, SDF, ZIP)
├── docs/screenshots/     # Screenshots für README
├── environment.yml       # Conda Environment
├── Dockerfile            # Docker Setup
├── README.md             # Hauptdokumentation
└── PROJECT.md            # Projektbeschreibung (Paper-Style)
```


## ⚡ Quickstart

### 1. Repository klonen
git clone git@github.com:clavinci94/drug-discovery-assistant.git
cd drug-discovery-assistant

### 2. Mit Conda starten
conda env create -f environment.yml
conda activate drug_discovery

python test_setup.py  # Setup-Test
export PYTHONPATH="."
GRADIO_SERVER_PORT=7860 python scripts/app_gradio.py
👉 Öffne im Browser: http://127.0.0.1:7860

### 3. Mit Docker starten
docker build -t drug-discovery:cpu .
PORT=7861 docker compose up
👉 Öffne im Browser: http://127.0.0.1:7861

## 📊 Ergebnisse
Random Forest: AUC ≈ 0.86
Cross-Attention TDI: AUC ≈ 0.87–0.88, PR-AUC > 0.93
Residue-Importance stimmt mit bekannten VEGFR2-Kinase-Domänen überein
Optimierte Moleküle zeigen verbesserte Aktivität + Drug-Likeness Scores

## ⚠️ Limitierungen
Demo basiert auf VEGFR2 (CHEMBL279), Single-Target
Datensatzgröße begrenzt (≤ 1000 Moleküle)
Keine experimentelle Validierung (nur Forschungs-Demo, kein Medizinprodukt)

## 📖 Weitere Infos
Eine ausführliche Projektbeschreibung mit Methodik, Ergebnissen und Limitierungen findest du in PROJECT.md.

## 👤 Autor
Claudio Vinci  
[LinkedIn](https://www.linkedin.com/in/claudio-vinci/) • [GitHub](https://github.com/clavinci94@gmail.com) • [Email](mailto:claudiovinci94)


## 📝 Lizenz
Dieses Projekt ist unter der MIT-Lizenz veröffentlicht.
