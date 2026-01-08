## Project Description: Heart Disease Prediction using SVM
This project focuses on developing a predictive model to identify individuals at risk of Coronary Heart Disease (CHD) using a Support Vector Machine (SVM) classifier. The dataset, CHDdata2.csv, contains various health and lifestyle factors.

## Key Steps:
1) Data Loading and Initial Inspection: Loaded the dataset and performed initial data exploration, including checking data types and descriptive statistics.
   
2) Data Preprocessing: Renamed columns for better readability and understanding.
   Introduced a synthetic 'sex' column for comprehensive analysis.
   Handled categorical features by applying LabelEncoder to the 'famhist' (family history) column.
   Checked for and confirmed the absence of missing values.
   
3) Feature Engineering and Scaling:
   Separated features (X) from the target variable (Y, CHD presence).
   Split the dataset into training and testing sets to ensure robust model evaluation.
   Applied StandardScaler to normalize numerical features, which is crucial for SVM performance.
   
4) Model Development:
Implemented a Linear Support Vector Machine (SVM) classifier.
Trained the SVM model on the scaled training data.

5) Model Evaluation:
Evaluated the model's performance using a classification report, providing metrics such as precision, recall, f1-score, and support.
Calculated and reported the overall accuracy of the SVM model.

6) Pipeline Creation and Persistence:
Developed a scikit-learn pipeline that encapsulates both the StandardScaler and the SVM model.
Saved the trained pipeline using joblib for easy deployment and future use.

This project demonstrates a complete machine learning workflow, from data preprocessing to model deployment, specifically tailored for a healthcare classification task. The use of a pipeline ensures that data transformations are consistently applied during both training and prediction phases.
