# Task 4: Classification with Logistic Regression

## Project Overview
This project implements a binary classification model using Logistic Regression and the Breast Cancer Wisconsin dataset from Scikit-learn.

The objective is to classify tumors as malignant or benign.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Steps
1. Loaded and explored the Breast Cancer Wisconsin dataset.
2. Checked dataset dimensions, class distribution and missing values.
3. Split the data into 80% training and 20% testing sets.
4. Standardized features using StandardScaler.
5. Trained a Logistic Regression model.
6. Evaluated the model using a confusion matrix, precision, recall and ROC-AUC.
7. Tested a lower classification threshold of 0.3.
8. Visualized the confusion matrix, ROC curve and sigmoid function.

## Model Results

| Metric | Result at threshold 0.5 |
|---|---:|
| Precision | 98.61% |
| Recall | 98.61% |
| ROC-AUC | 99.54% |

## Threshold Tuning

At threshold 0.3:
- Precision: 97.30%
- Recall: 100%
- False negatives: 0
- False positives: 2

Lowering the threshold increased recall but reduced precision.

## Visualizations
- `confusion_matrix.png`
- `roc_curve.png`
- `sigmoid_function.png`

## Project Files
- `logistic_regression.py` – Python implementation.
- `README.md` – Project documentation.

## Conclusion
The Logistic Regression model achieved strong results on the test dataset. Threshold tuning demonstrated the trade-off between precision and recall.

This is an educational project, not a clinically validated diagnostic system.