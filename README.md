# SMLSML Credit Scoring

Workspace submission untuk **Rafli Ardiansyah** yang disiapkan agar memenuhi target penilaian Advanced pada proyek Dicoding SMSML.

Nama dashboard Grafana yang digunakan sebagai bukti monitoring:

```text
dashboard-mikachuu
```

Preferensi environment lokal:

- Python: `3.12.7`
- Pengelola environment/dependency: `uv`
- MLflow: `2.19.0`
- Runtime container: `podman`

## Alur Lokal

```bash
uv venv --python 3.12.7
uv pip install -r requirements.txt
python Eksperimen_SML_Rafli-Ardiansyah/preprocessing/automate_Rafli_Ardiansyah.py
mlflow ui --host 127.0.0.1 --port 5000
python SMSML_Rafli-Ardiansyah/Membangun_model/modelling.py
python SMSML_Rafli-Ardiansyah/Membangun_model/modelling_tuning.py
```

Pastikan kredensial DagsHub sudah tersedia sebelum menjalankan `modelling_tuning.py`, karena file tersebut mengirimkan tracking run, artifact, dan model registry ke DagsHub.

## Repository Target

```text
https://github.com/bulgogipedas/smlsml-credit-scoring
```
