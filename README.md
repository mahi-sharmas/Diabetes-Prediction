# Diabetes Prediction Using Machine Learning

A comparative study of classification algorithms for predicting diabetes onset using the Pima Indians Diabetes dataset. Implements Logistic Regression, SVM, and KNN with hyperparameter tuning via Grid Search to identify the best-performing model.

## Problem Statement

Early detection of diabetes can significantly improve patient outcomes. This project builds and compares multiple ML classifiers to predict whether a patient is diabetic based on diagnostic health measurements, with a focus on finding the optimal model through systematic hyperparameter tuning.

## Dataset

- **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features (8):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target:** `Outcome` (1 = Diabetic, 0 = Non-Diabetic)
- **Class Split:** 65% Non-Diabetic, 35% Diabetic

## Approach

1. **Preprocessing** — Replaced biologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` with NaN, then imputed using median values
2. **Feature Scaling** — Applied `StandardScaler` for distance-based models (SVM, KNN)
3. **Train/Test Split** — 80/20 split with `random_state=42`
4. **Model Training** — Logistic Regression, SVM (RBF kernel), KNN (k=5)
5. **Hyperparameter Tuning** — Grid Search with 5-fold CV on SVM (`C`, `kernel`, `gamma`) and KNN (`n_neighbors`, `weights`, `metric`)

## Key Results

| Model | Accuracy | Precision (Diabetic) | Recall (Diabetic) | F1 (Diabetic) |
|---|---|---|---|---|
| Logistic Regression | 0.75 | 0.67 | 0.62 | 0.64 |
| SVM (default RBF) | 0.75 | 0.67 | 0.58 | 0.62 |
| KNN (k=5) | 0.72 | 0.60 | 0.67 | 0.63 |
| **SVM (Tuned — linear, C=1)** | **0.75** | **0.67** | **0.62** | **0.64** |
| KNN (Tuned — manhattan, k=7) | 0.73 | 0.61 | 0.65 | 0.63 |

- **Best Model:** SVM with linear kernel (C=1, gamma=scale) — 75% accuracy, best precision-recall balance
- **Grid Search CV Score:** 0.77 (SVM), 0.77 (KNN)

## Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib
- **Techniques:** Logistic Regression, SVM, KNN, Grid Search CV, StandardScaler, Confusion Matrix Analysis

## Project Structure

```
├── Diabetes_Prediction.ipynb   # Full analysis and modeling notebook
├── archive.zip                 # Dataset (Pima Indians Diabetes)
├── diabetes-prediction.jpeg    # Visualization
└── README.md
```

## How to Run

```bash
unzip archive.zip
jupyter notebook Diabetes_Prediction.ipynb
```

## Author

**Mahi Sharma**
B.Tech CSE (Data Science) — Manipal University Jaipur
[GitHub](https://github.com/mahi-sharmas)
