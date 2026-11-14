# Cardio Risk Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green)
![pandas](https://img.shields.io/badge/pandas-data-informational)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

Heart disease risk prediction on 918 real patient records. This project trains a Decision Tree, a Random Forest and an XGBoost model on clinical measurements, studies how each one overfits, and compares them on how well they find patients who actually have heart disease.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Quality Findings](#data-quality-findings)
- [Methodology](#methodology)
- [Models and Hyperparameters](#models-and-hyperparameters)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

## Overview

Cardiovascular disease is the leading cause of death worldwide, and early detection is a large part of managing it. Given a patient's routine clinical measurements, the goal here is to predict whether that patient has heart disease. This is a binary classification problem (1 = heart disease, 0 = normal).

The project has three goals:

1. Build and tune three tree-based models on a real medical dataset
2. Understand overfitting by comparing training accuracy against validation accuracy for each hyperparameter
3. Evaluate the models with metrics that matter in a medical setting, not just accuracy

## Dataset

- Source: Heart Failure Prediction Dataset on Kaggle, by fedesoriano. Check the dataset page for its license and terms before reusing the data.
- Size: 918 patients, 11 input features and 1 target column
- Class balance: 508 patients with heart disease (55%) and 410 without (45%), so the classes are fairly balanced
- Missing values: none recorded as blanks, but some are hidden as zeros (see Data Quality Findings)

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

After one-hot encoding the 5 categorical columns, the model sees 20 input features.

## Data Quality Findings

| Issue | Count | Why it matters | Handling |
|---|---|---|---|
| `Cholesterol` recorded as 0 | 172 of 918 patients | A cholesterol of 0 is not a real reading, it is a placeholder for missing data. These patients also have an 88% heart disease rate, against 48% for everyone else, so the zeros carry signal and cannot be ignored | Decision pending: documented in the notebook |
| `RestingBP` recorded as 0 | 1 patient | Impossible value | Row removed |
| Negative `Oldpeak` | 13 patients | Plausible clinically (ST elevation), so these are kept | Kept as is |
| Duplicate rows | 0 | None found | None needed |
| Sex imbalance | 725 male, 193 female | Heart disease rate is 63% in the male rows and 26% in the female rows, so the model learns from a skewed sample | Listed under Limitations |

## Methodology

1. **Load and inspect.** Read the CSV with pandas and check shape, columns and value ranges.
2. **Clean.** Fix the impossible and placeholder values described above.
3. **Encode.** One-hot encode the 5 categorical columns with `pd.get_dummies`, which turns each category into its own 0/1 column.
4. **Split.** An 80/20 train/validation split (734 training rows, 184 validation rows) with a fixed random seed of 55 so results are reproducible.
5. **Tune.** For each model, vary one hyperparameter at a time and plot training accuracy against validation accuracy. The point where the two lines separate is where the model starts to overfit.
6. **Train final models.** Combine the best value of each hyperparameter into one model per algorithm.
7. **Early stopping for XGBoost.** The training set is further split 80/20 into a fit part (587 rows) and an eval part (147 rows). XGBoost scores itself on the eval part after every tree and stops once 10 rounds in a row fail to improve. The validation set is never used for this decision.
8. **Evaluate.** Compare train and validation accuracy, then add the medical-relevant metrics (see Roadmap).

## Models and Hyperparameters

| Model | Hyperparameters tuned | Final values |
|---|---|---|
| Decision Tree | `max_depth`, `min_samples_split` | max_depth 3, min_samples_split 50 |
| Random Forest | `max_depth`, `min_samples_split`, `n_estimators` | 100 trees, max_depth 16, min_samples_split 10 |
| XGBoost | `learning_rate`, `n_estimators` with early stopping | learning rate 0.1, up to 500 trees, early_stopping_rounds 10, best round 17 |

- **Decision Tree:** a single tree that splits the data using the features that reduce impurity the most. Easy to read, but it overfits quickly when it grows deep.
- **Random Forest:** many trees, each trained on a random sample of the rows and a random subset of the features. The trees vote, which cancels out much of the overfitting.
- **XGBoost:** trees trained one after another, where each new tree is fitted to correct the mistakes of the ones before it.

## Results

Preliminary numbers from the current notebook. They come before the data cleaning and cross-validation steps in the Roadmap, so they will change.

| Model | Train accuracy | Validation accuracy |
|---|---|---|
| Decision Tree | 0.8583 | 0.8641 |
| Random Forest | 0.9305 | 0.8913 |
| XGBoost | 0.9319 | 0.8533 |

Recall, precision, F1 and ROC-AUC will be added once those cells are built.

## Key Takeaways

- The Random Forest scored best on validation in this run.
- XGBoost did not beat the single Decision Tree here, even though it trains to a higher training accuracy. It shows the largest gap between training and validation accuracy.
- The validation set has only 184 patients, so each patient is worth about 0.5% accuracy. A difference of one or two patients between models is noise, not a real ranking.
- A shallow tree (depth 3) with no signs of overfitting got close to the ensembles, which shows that more complex models are not automatically better on a small dataset.

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
└── models/
    └── (saved trained model, added in a later step)
```

## Getting Started

Requirements: Python 3.x and Jupyter Notebook.

```bash
git clone https://github.com/Abdullahjaved-82/cardio-risk-predictor.git
cd cardio-risk-predictor
pip install -r requirements.txt
jupyter notebook notebooks/
```

Open `heart_disease_prediction.ipynb` and run the cells from top to bottom. The notebook reads the data from the `data/` folder, and all random seeds are fixed, so you should get the same numbers.

## Limitations

- The dataset is small (918 patients), so results are noisy and may not hold on new data.
- About 79% of the patients are male, so the model may perform worse for female patients.
- Hyperparameters were picked by looking at validation scores, so the validation numbers are slightly optimistic.
- The data comes from a single public source and says nothing about where or how it was collected.
- This is a learning and portfolio project. It is not a medical tool and must not be used for diagnosis or treatment decisions.

## Roadmap

- [x] Decision Tree, Random Forest and XGBoost trained and tuned
- [x] Early stopping for XGBoost on a held-out eval set
- [ ] Data cleaning (placeholder cholesterol values, impossible blood pressure)
- [ ] Exploratory analysis section
- [ ] Confusion matrix, precision, recall, F1 and ROC-AUC for all three models
- [ ] 5-fold cross-validation for more reliable numbers
- [ ] Feature importance plot
- [ ] Save the best model and predict on a new patient

## Acknowledgements

- Dataset by fedesoriano on Kaggle
- The tree ensemble workflow follows what I learned in Andrew Ng's Machine Learning Specialization (DeepLearning.AI). My other notebooks from that course are in [ml-specialization-notes](https://github.com/Abdullahjaved-82/ml-specialization-notes).

## Author

Abdullah Javed, Software Engineering student at COMSATS University Islamabad.

- LinkedIn: [linkedin.com/in/abdullah-javed-8468a7343](https://www.linkedin.com/in/abdullah-javed-8468a7343)
- Portfolio: [abdullahjaved.site](https://abdullahjaved.site)
- Email: abdullahjavec@gmail.com
