## Diabetes Prediction Using Machine Learning

A comparative study of classification algorithms to predict diabetes onset from clinical measurements, with hyperparameter tuning via GridSearchCV to optimize model performance.

### Highlights

- Compared 3 classifiers (Logistic Regression, SVM, KNN) with systematic hyperparameter tuning using 5-fold cross-validation
- Achieved 75% test accuracy with a tuned SVM (linear kernel, C=1) — the best-performing model with 77% cross-validation score
- Addressed biologically impossible zero values in 5 features through median imputation before training
- Built a complete pipeline from preprocessing through evaluation with confusion matrix visualizations

### Problem Statement

Early detection of diabetes is critical for effective medical intervention and patient outcomes. This project uses the Pima Indians Diabetes Database — containing 8 clinical features from 768 patients — to build and compare multiple ML classifiers. The goal is to identify which algorithm generalizes best on unseen data after proper preprocessing and hyperparameter tuning.

### Dataset

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) (Kaggle)
- **Size:** 768 patients × 9 columns (8 features + 1 target)
- **Target:** `Outcome` (binary — 1 = Diabetic, 0 = Non-Diabetic)
- **Class distribution:** 65% Non-Diabetic (500), 35% Diabetic (268)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Data quality issue:** Biologically impossible zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI — handled via median imputation

### Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![NumPy](https://img.shields.io/badge/NumPy-Compute-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-teal)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-red)

### Methodology

1. **Data Loading & EDA** — Imported the dataset and visualized the class distribution (65/35 split) using countplots
2. **Data Cleaning** — Identified and replaced biologically impossible zeros in 5 features with NaN, then imputed using column medians
3. **Feature Scaling** — Applied StandardScaler to normalize all 8 features for distance-based models (SVM, KNN)
4. **Train-Test Split** — 80/20 split with `random_state=42` (614 train, 154 test samples)
5. **Baseline Training** — Trained Logistic Regression, SVM (RBF kernel), and KNN (k=5) as baseline models
6. **Hyperparameter Tuning** — Used GridSearchCV with 5-fold cross-validation to tune SVM (C: [0.1, 1, 10], kernel: [linear, rbf, poly], gamma: [scale, auto]) and KNN (n_neighbors: [3, 5, 7, 9, 11, 13], weights: [uniform, distance], metric: [euclidean, manhattan])
7. **Evaluation** — Compared all 5 models on accuracy, precision, recall, F1-score, and confusion matrix heatmaps

### Key Results

| Model | Test Accuracy | CV Score | Precision (Diabetic) | Recall (Diabetic) | F1 (Diabetic) |
|---|---|---|---|---|---|
| Logistic Regression | 75% | — | 0.67 | 0.62 | 0.64 |
| SVM (Baseline, RBF) | 75% | — | 0.67 | 0.58 | 0.62 |
| KNN (Baseline, k=5) | 72% | — | 0.60 | 0.67 | 0.63 |
| **SVM (Tuned, Linear, C=1)** | **75%** | **77%** | **0.67** | **0.62** | **0.64** |
| KNN (Tuned, k=7, Manhattan) | 73% | 77% | 0.61 | 0.65 | 0.63 |

**Best model:** Tuned SVM with `C=1, kernel='linear', gamma='scale'` — 75% test accuracy with the best precision-recall balance on the diabetic class. Both tuned models achieved 77% on cross-validation, indicating good generalization.

### How to Run

```bash
git clone https://github.com/mahi-sharmas/Diabetes-Prediction.git
cd Diabetes-Prediction
pip install -r requirements.txt
jupyter notebook Diabetes_Prediction.ipynb
```

### Project Structure

```
Diabetes-Prediction/
├── Diabetes_Prediction.ipynb    # Full ML pipeline — EDA, preprocessing, training, tuning, evaluation
├── archive.zip                  # Compressed Pima Indians Diabetes dataset
├── diabetes-prediction.jpeg     # Project visualization
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Future Improvements

- Add ensemble methods (Random Forest, XGBoost, Gradient Boosting) to push accuracy beyond 75%
- Implement SHAP or LIME for model interpretability to identify which clinical features drive diabetes predictions
- Build a Flask or Streamlit web interface for real-time patient risk assessment

### Author

**Mahi Sharma** — B.Tech CSE (Data Science), Manipal University Jaipur (2023–2027)

GitHub: [github.com/mahi-sharmas](https://github.com/mahi-sharmas) | Email: mahi.sh4rma7@gmail.com
