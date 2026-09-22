# German Credit Dataset — Predictive Modelling Baseline

## Objective

Build a **reproducible, research-grade predictive modelling baseline** on the German Credit dataset, targeting **80%+ prediction accuracy** with scientifically correct evaluation methodology.

This baseline will later feed into a separate **credit fairness and bias analysis pipeline**.

## Dataset

The [German Credit dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) from UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Source | Prof. Dr. Hans Hofmann, Universität Hamburg |
| Instances | 1,000 |
| Features | 20 (7 numerical, 13 categorical) |
| Target | Credit risk: 1 = Good, 2 = Bad |
| Class Distribution | ~70% Good, ~30% Bad |

## Models

Four primary models are compared:

| Model | Type | Library |
|---|---|---|
| XGBoost | Gradient Boosting | `xgboost` |
| CatBoost | Categorical-aware Boosting | `catboost` |
| Random Forest | Bagging | `scikit-learn` |
| MLP | Neural Network | `scikit-learn` |

## Experimental Methodology

```
Raw Dataset → EDA → 80/20 Stratified Split
                        ↓
              80% Development Set
              (5-Fold Stratified CV)
                        ↓
              Hyperparameter Tuning
                        ↓
              Best Hyperparameters
                        ↓
              Retrain on Full 80%
                        ↓
              Final 20% Test Evaluation
```

### Critical Rules

- **80/20 stratified split** with fixed random seed (42)
- **5-fold stratified cross-validation** on the development set for all model selection
- The **20% test set remains completely untouched** until final evaluation in Notebook 04
- No data leakage: preprocessing fitted only on training portions
- Primary optimization metric: **Accuracy**
- Secondary metrics: ROC-AUC, Precision, Recall, F1, Specificity

## Project Structure

```
├── data/
│   ├── german.data          # Original dataset (space-separated, no header)
│   └── german.doc           # Dataset documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA and data understanding
│   ├── 02_model_training.ipynb         # Baseline models with 5-fold CV
│   ├── 03_hyperparameter_tuning.ipynb  # Systematic hyperparameter optimization
│   └── 04_final_model_evaluation.ipynb # Final test-set evaluation
│
├── results/
│   ├── tables/              # CSV tables with metrics and comparisons
│   ├── plots/               # Visualization outputs (PNG)
│   └── predictions/         # Fairness-ready prediction files (CSV)
│
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter nbconvert --execute notebooks/01_data_exploration.ipynb
jupyter nbconvert --execute notebooks/02_model_training.ipynb
jupyter nbconvert --execute notebooks/03_hyperparameter_tuning.ipynb
jupyter nbconvert --execute notebooks/04_final_model_evaluation.ipynb
```

Or open them in Jupyter Lab / Notebook and run cells sequentially.

**Important**: Run notebooks in order (01 → 02 → 03 → 04). Notebook 04 depends on hyperparameter results saved by Notebook 03.

## Results

After execution:

- **`results/tables/`** — Baseline and tuned CV results, final comparison table
- **`results/plots/`** — Model comparison charts, confusion matrices, ROC curves
- **`results/predictions/`** — Per-model prediction files with original features for fairness analysis

### Prediction File Format

Each file in `results/predictions/` contains:

| Column | Description |
|---|---|
| sample_index | Original dataset index |
| true_label | Actual credit risk (1=Good, 0=Bad) |
| predicted_label | Model prediction |
| predicted_probability | P(Good Credit) |
| All original features | Including protected attributes (personal_status, age, foreign_worker) |

## Reproducibility

- Fixed random seed: **42**
- Stratified splits preserve class distribution
- All hyperparameter search spaces documented
- Best hyperparameters saved to `results/tables/best_hyperparameters.json`

## License

Dataset provided by Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg.
