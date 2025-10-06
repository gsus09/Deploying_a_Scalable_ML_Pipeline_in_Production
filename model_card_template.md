# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model addresses a binary classification task: predicting whether an individual earns more than $50,000 per year. It is implemented using a GradientBoostingClassifier from scikit-learn 1.7.2, with hyperparameters optimized for performance.

## Intended Use
The model is intended for educational, academic, and research purposes. It can be used to explore classification techniques, feature engineering, and model evaluation in the context of income prediction based on demographic and employment-related attributes.

## Training Data
The model was trained on the Census Income Dataset from the UCI Machine Learning Repository. The dataset contains 32,561 rows and 15 columns, including:

- Target variable: salary (binary: <=50K, >50K)
- Features: 8 categorical and 6 numerical attributes

Basic preprocessing included:

- Removal of leading/trailing whitespaces
- One-hot encoding of categorical features
- Label binarization of the target variable

An 80/20 train-test split was applied, with stratification on the target label to preserve class distribution.

## Evaluation Data
The remaining 20% of the dataset was reserved for evaluation. The same preprocessing steps (using encoders fitted on the training set) were applied to ensure consistency.

## Metrics
Model performance was assessed using standard classification metrics:

- Precision: 0.75
- Recall: 0.60
- Fβ Score (β=1): 0.67

These metrics reflect a balanced trade-off between false positives and false negatives, with a slight emphasis on precision.

## Ethical Considerations
This model is trained on historical census data, which may reflect biases present in the original dataset. It should not be used for real-world decision-making or to infer income levels for individuals or groups. The dataset does not represent a fair or current view of income distribution across populations.

## Caveats and Recommendations
- The dataset originates from the 1994 U.S. Census, making it outdated for modern socioeconomic analysis.
- It is not representative of current population demographics or labor market conditions.
- Recommended use is strictly for educational or experimental purposes, such as testing classification pipelines or exploring fairness in machine learning.
