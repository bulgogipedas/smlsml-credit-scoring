# SMLSML Credit Scoring

Submission workspace for **Rafli Ardiansyah** targeting the advanced rubric for the Dicoding SMSML project.

Dashboard evidence must use this Grafana dashboard name:

```text
dashboard-mikachuu
```

Local tooling preference:

- Python: `3.12.7`
- Dependency runner: `uv`
- MLflow: `2.19.0`
- Container runtime: `podman`

## Local flow

```bash
uv venv --python 3.12.7
uv pip install -r requirements.txt
python Eksperimen_SML_Rafli-Ardiansyah/preprocessing/automate_Rafli_Ardiansyah.py
mlflow ui --host 127.0.0.1 --port 5000
python SMSML_Rafli-Ardiansyah/Membangun_model/modelling.py
python SMSML_Rafli-Ardiansyah/Membangun_model/modelling_tuning.py
```

Use DagsHub credentials before running `modelling_tuning.py` for the advanced online tracking requirement.

## Repository target

```text
https://github.com/bulgogipedas/smlsml-credit-scoring
```
