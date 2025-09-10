# Schnell & reproducible: Micromamba + conda-forge
FROM mambaorg/micromamba:1.5.8

# Arbeitsverzeichnis
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Systemabhängige Tools (curl für fair-esm Model-Downloads via torch hub cache)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Env-Dateien zuerst (bessere Docker-Layer-Caches)
COPY environment.yml /app/environment.yml

# Conda-Env erstellen (rdkit, pytorch CPU, numpy, pandas, ... kommt aus environment.yml)
RUN micromamba create -y -n drug_discovery -f /app/environment.yml && \
    micromamba clean --all --yes

# Projektcode kopieren
COPY . /app

# Zusatz-Pip-Pakete (falls noch nicht in environment.yml)
# fair-esm (ESM-2), gradio (UI), optuna (HPO)
RUN micromamba run -n drug_discovery pip install --upgrade fair-esm gradio optuna

# Port für Gradio
EXPOSE 7860

# Default-Start: Gradio-App
CMD ["/bin/bash","-lc","micromamba run -n drug_discovery python scripts/app_gradio.py"]
