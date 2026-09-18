# Drug Classification — End-to-End MLOps Pipeline

A small multi-class classification problem wrapped in a complete MLOps chain:
reproducible training, experiment tracking, a hyperparameter sweep, a model registry,
a served REST + web API, containerisation, and metric monitoring in Elasticsearch/Kibana.

**The model is deliberately simple. The pipeline around it is the point.** The dataset is
a 200-row teaching set, so nothing here is a modelling achievement — what the repository
demonstrates is the path from a commit to a running, tracked, monitored endpoint.

---

## Results

Logistic Regression, stratified 80/20 split, `random_state=42` — reproducible with
`make train`.

| Metric | Value |
|---|---|
| Accuracy | **97.50 %** |
| Weighted F1 | 0.97 |
| Macro F1 | 0.99 |
| Test set size | 40 samples |

Per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DrugY | 0.95 | 1.00 | 0.97 | 18 |
| drugA | 1.00 | 1.00 | 1.00 | 5 |
| drugB | 1.00 | 1.00 | 1.00 | 3 |
| drugC | 1.00 | 1.00 | 1.00 | 3 |
| drugX | 1.00 | 0.91 | 0.95 | 11 |

Confusion matrix — a single misclassification, one `drugX` predicted as `DrugY`:

```
        DrugY  drugA  drugB  drugC  drugX
DrugY      18      0      0      0      0
drugA       0      5      0      0      0
drugB       0      0      3      0      0
drugC       0      0      0      3      0
drugX       1      0      0      0     10
```

### Read these numbers carefully

**97.50 % is 39 correct out of 40.** On a test set that small, one sample is worth 2.5
points of accuracy, so the confidence interval is wide and this figure should not be
compared against results on real datasets. It is reported here because it is what the
pipeline produces, not because it demonstrates modelling skill.

## Known limitations

- **Test set of 40 samples.** No cross-validation, single split — the accuracy has high
  variance.
- **Features are not scaled.** `Age` and `Na_to_K` are numeric and unscaled alongside
  label-encoded categoricals, and `LogisticRegression` emits a convergence warning at
  `max_iter=1000`. Adding a `StandardScaler` in the pipeline is the correct fix.
- **Label encoders are not persisted.** The API re-encodes features by hand in
  `encode_features()`, which would drift if the training encoding changed. A saved
  `ColumnTransformer` is the robust answer.
- **`drug200.csv` is a teaching dataset** — balanced, clean, and not representative of a
  production classification problem.

---

## What the pipeline actually does

| Stage | Implementation |
|---|---|
| Data preparation | `model_pipeline.py` — label encoding, stratified split, fixed seed |
| Training | Logistic Regression, reproducible from `main.py --step train` |
| Experiment tracking | `mlflow_config.py` — params, metrics, dataset info, model registry |
| Hyperparameter search | `hyperparameter_tuning.py` — 5 `C` × 4 solvers × 3 `max_iter` = 60 tracked runs |
| Monitoring | `train_with_monitoring.py` + `elastic_logger.py` — metrics shipped to Elasticsearch, visualised in Kibana |
| Serving | `app.py` — FastAPI, REST endpoint plus a Jinja2 web form |
| Containerisation | `Dockerfile` (API) and `docker-compose.yml` (Elasticsearch + Kibana) |
| Automation | `Makefile` — `install`, `data`, `train`, `evaluate`, `predict`, `mlflow`, `clean`, `lint` |

## Run it

```bash
pip install -r requirements.txt

make train            # prepare, train, evaluate, save the model
make evaluate         # score the saved model
```

### With experiment tracking

```bash
make mlflow           # MLflow server on :5000 (SQLite backend)
make run              # a tracked training run
python hyperparameter_tuning.py   # the 60-run sweep
```

### With monitoring

```bash
docker compose up -d              # Elasticsearch :9200 + Kibana :5601
python train_with_monitoring.py   # metrics to MLflow and Elasticsearch
```

### Serving

```bash
uvicorn app:app --reload          # http://localhost:8000
# or
docker build -t drug-api . && docker run -p 8000:8000 drug-api
```

## Repository layout

```
model_pipeline.py          data preparation, training, evaluation, persistence
model_pipeline_mlflow.py   the same pipeline, MLflow-instrumented
main.py                    CLI entry point (--step all|train|evaluate|predict)
mlflow_config.py           experiment setup, dataset and model logging
hyperparameter_tuning.py   tracked grid search
train_with_monitoring.py   training with MLflow + Elasticsearch
elastic_logger.py          Elasticsearch metrics logger
app.py                     FastAPI service (REST + web form)
templates/                 Jinja2 templates
Dockerfile                 API image
docker-compose.yml         Elasticsearch + Kibana
Makefile                   task automation
data/drug200.csv           dataset (200 rows, 5 classes)
```

## Dataset

`drug200.csv` — 200 patients, 5 features (`Age`, `Sex`, `BP`, `Cholesterol`, `Na_to_K`),
target `Drug` with 5 classes.
