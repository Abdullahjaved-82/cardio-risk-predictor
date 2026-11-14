# Cardio Risk Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)
![pandas](https://img.shields.io/badge/pandas-data-informational)

Heart disease risk prediction on 918 real patient records. This project cleans a clinical dataset, trains a Decision Tree, a Random Forest and an XGBoost model, studies how each one overfits, and compares them on the metrics that matter in a medical setting, including how many sick patients each model actually finds. The best model is saved and can predict for a new patient.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Cleaning](#data-cleaning)
- [Methodology](#methodology)
- [Models and Hyperparameters](#models-and-hyperparameters)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Predicting for a New Patient](#predicting-for-a-new-patient)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

## Overview

Cardiovascular disease is the leading cause of death worldwide, and early detection is a large part of managing it. Given a patient's routine clinical measurements, the goal is to predict whether the patient has heart disease. This is a binary classification problem (1 = heart disease, 0 = normal).

The project has three goals:

1. Build and tune three tree-based models on a real medical dataset
2. Understand overfitting by comparing training accuracy against validation accuracy for each hyperparameter
3. Evaluate the models with recall, precision, F1 and ROC-AUC, not just accuracy

## Dataset

- Source: Heart Failure Prediction Dataset on Kaggle, by fedesoriano. Check the dataset page for its license and terms before reusing the data.
- Size: 918 patients, 11 input features and 1 target column
- Class balance: 508 patients with heart disease (55%) and 410 without (45%)

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Age of the patient in years |
| Sex | Categorical | M or F |
| ChestPainType | Categorical | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| RestingBP | Numeric | Resting blood pressure in mm Hg |
| Cholesterol | Numeric | Serum cholesterol in mg/dl |
| FastingBS | Binary | 1 if fasting blood sugar is above 120 mg/dl, otherwise 0 |
| RestingECG | Categorical | Normal, ST (ST-T wave abnormality) or LVH (probable left ventricular hypertrophy) |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Y or N, whether exercise caused angina |
| Oldpeak | Numeric | ST depression measured during exercise |
| ST_Slope | Categorical | Slope of the peak exercise ST segment: Up, Flat or Down |
| HeartDisease | Target | 1 = heart disease, 0 = normal |

After cleaning and one-hot encoding, the models see 21 input features.

## Data Cleaning

| Issue | Count | Why it matters | Handling |
|---|---|---|---|
| `Cholesterol` recorded as 0 | 172 of 918 patients | A cholesterol of 0 is a placeholder for a missing reading, not a real value. These patients also have an 88% heart disease rate, against 48% for everyone else, so the zeros carry signal | A new column `Cholesterol_missing` flags them, then the zeros are replaced with the median of the real readings (237) |
| `RestingBP` recorded as 0 | 1 patient | Impossible value | Row removed, leaving 917 patients |
| Negative `Oldpeak` | 13 patients | Plausible clinically (ST elevation) | Kept as is |
| Duplicate rows | 0 | None found | None needed |
| Sex imbalance | 725 male, 193 female | Heart disease rate is 63% in the male rows and 26% in the female rows | Listed under Limitations |

## Methodology

1. **Load and inspect** the data with pandas.
2. **Clean** the placeholder and impossible values described above.
3. **Encode** the 5 categorical columns with `pd.get_dummies`, which turns each category into its own 0/1 column.
4. **Split** 80/20 into 733 training rows and 184 validation rows, with a fixed random seed of 55 so results are reproducible.
5. **Tune** each model by varying one hyperparameter at a time and plotting training against validation accuracy. The point where the two lines separate is where the model starts to overfit.
6. **Train final models** using the best value of each hyperparameter.
7. **Stop XGBoost early.** A small eval set is split out of the training data, and XGBoost stops once 10 rounds in a row fail to improve on it. The validation set is never used for this decision.
8. **Cross-validate.** 5-fold stratified cross-validation on all the cleaned data gives a more stable score than one small validation split.
9. **Evaluate** with accuracy, precision, recall, F1, ROC-AUC and confusion matrices.
10. **Save** the model with the best validation recall using joblib.

## Models and Hyperparameters

| Model | Hyperparameters tuned | Final values |
|---|---|---|
| Decision Tree | `max_depth`, `min_samples_split` | max_depth 4, min_samples_split 30 |
| Random Forest | `max_depth`, `min_samples_split`, `n_estimators` | 100 trees, max_depth 8, min_samples_split 10 |
| XGBoost | `learning_rate`, `n_estimators` with early stopping | learning rate 0.1, up to 500 trees, early stopping after 10 rounds without improvement |

- **Decision Tree:** a single tree that splits the data using the features that reduce impurity the most. Easy to read, but it overfits quickly when it grows deep.
- **Random Forest:** many trees, each trained on a random sample of the rows and a random subset of the features. The trees vote, which cancels out much of the overfitting.
- **XGBoost:** trees trained one after another, where each new tree is fitted to correct the mistakes of the ones before it.

The forest uses depth 8 rather than 16 because both scored the same on validation, and the shallower forest is simpler with a smaller train-to-validation gap.

## Results

Validation set (184 patients) unless stated otherwise.

| Model | Train accuracy | Val accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV accuracy |
|---|---|---|---|---|---|---|---|
| Decision Tree | 0.8772 | 0.8859 | 0.9167 | 0.8919 | 0.9041 | 0.9196 | 0.8332 |
| Random Forest | 0.9236 | 0.9022 | 0.9115 | 0.9279 | 0.9196 | 0.9399 | 0.8757 |
| XGBoost | 0.9509 | 0.8859 | 0.9018 | 0.9099 | 0.9058 | 0.9461 | 0.8582 |

Plots are saved in the `images/` folder: the tuning curves for each hyperparameter, the confusion matrices for the three models, and the random forest feature importance.

## Key Takeaways

- The Random Forest has the best validation accuracy, recall and F1, and the best cross-validation accuracy. XGBoost has the highest ROC-AUC.
- The differences between the models are small. The validation set has 184 patients, so each patient is worth about 0.5% accuracy, and a gap of one or two patients is noise, not a real ranking.
- Recall is the metric to watch here, because it is the share of sick patients the model finds. The Random Forest finds about 93% of them.
- Cross-validation tells a steadier story than the single split: the Random Forest is ahead and the single Decision Tree is last.
- The Random Forest relies most on the ST segment slope, the ST depression (Oldpeak), exercise-induced angina, chest pain type and maximum heart rate.

## Predicting for a New Patient

The notebook saves the best model to `models/heart_disease_model.joblib`, together with the feature list and the cholesterol median, so a new patient can be prepared in exactly the same way as the training data.

```python
import joblib
import pandas as pd

saved = joblib.load("models/heart_disease_model.joblib")

patient = pd.DataFrame([{
    'Age': 54, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 140,
    'Cholesterol': 239, 'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 120,
    'ExerciseAngina': 'Y', 'Oldpeak': 1.5, 'ST_Slope': 'Flat',
}])

patient['Cholesterol_missing'] = (patient['Cholesterol'] == 0).astype(int)
patient['Cholesterol'] = patient['Cholesterol'].replace(0, saved['cholesterol_median'])
patient = pd.get_dummies(patient, dtype=int).reindex(columns=saved['features'], fill_value=0)

print(saved['model'].predict_proba(patient)[0, 1])
```

This is a demo on made-up values and not a diagnosis.

## Repository Structure

```
cardio-risk-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── heart.csv
├── notebooks/
│   └── heart_disease_prediction.ipynb
├── images/
│   └── (tuning curves, confusion matrices, feature importance)
└── models/
    └── heart_disease_model.joblib
```

## Getting Started

Requirements: Python 3.x and Jupyter Notebook.

```bash
git clone https://github.com/Abdullahjaved-82/cardio-risk-predictor.git
cd cardio-risk-predictor
pip install -r requirements.txt
jupyter notebook notebooks/
```

Open `heart_disease_prediction.ipynb` and run the cells from top to bottom. The data is read from the `data/` folder and all random seeds are fixed, so you should get the same numbers. XGBoost 2.0 or newer is needed.

## Limitations

- The dataset is small (918 patients), so results are noisy and may not hold on new data.
- About 79% of the patients are male, so the model may perform worse for female patients.
- Hyperparameters were picked by looking at validation scores, so the validation numbers are slightly optimistic. The cross-validation scores share this weakness.
- The cholesterol median is computed on all rows before the train/validation split, a small leak that has a negligible effect at this dataset size.
- The data comes from a single public source and says nothing about where or how it was collected.
- This is a learning and portfolio project. It is not a medical tool and must not be used for diagnosis or treatment decisions.

## Acknowledgements

- Dataset by fedesoriano on Kaggle
- The tree ensemble workflow follows what I learned in Andrew Ng's Machine Learning Specialization (DeepLearning.AI). My other notebooks from that course are in [ml-specialization-notes](https://github.com/Abdullahjaved-82/ml-specialization-notes).

## Author

Abdullah Javed, Software Engineering student at COMSATS University Islamabad.

- LinkedIn: [linkedin.com/in/abdullah-javed-8468a7343](https://www.linkedin.com/in/abdullah-javed-8468a7343)
- Portfolio: [abdullahjaved.site](https://abdullahjaved.site)
- Email: abdullahjavec@gmail.com
