# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
!pip install lime
import lime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# SUPPORT VECTOR MACHINE ( MODEL )
from sklearn import svm

from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
)

# APPLY SMOTE (Synthetic Minority Over-sampling Technique)

!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Step 1: Load the dataset
heart = pd.read_csv("CHDdata2.csv")
heart.head(5)

# Rename the columns for better understanding
heart.rename(columns={
    'sbp': 'systolic_bp', 'tobacco': 'tobacco_per_yer',
    'ldl': 'chol','adiposity': 'adiposity',
    'famhist': 'famhist','typea': 'stress_prone','obesity': 'Bmi',
    'alcohol': 'alco_per_year','age': 'age',
    'chd': 'target'
}, inplace=True)

heart.head()


# Randomly assign sex: 1 = male, 0 = female
np.random.seed(42)  # Ensures reproducibility
heart['sex'] = np.random.choice([1, 0], size=len(heart), p=[0.6, 0.4])

heart.head()

columns = ['sex'] + [col for col in heart.columns if col != 'sex']
heart = heart[columns]
heart.head()

# Check Data Information
heart.info()

# Description of each column
heart.describe()


# Step 4: Check for Missing Data
heart.isnull().sum()

# Handle categorical features
# The labelencoder transform categorical labels into numerical labels.
encoder = LabelEncoder()
heart['famhist'] = encoder.fit_transform(heart['famhist'])
heart['famhist'].head()
heart.head()

# Feature selection
# Y is the dependent variable
# X is the independent variable
## Preparing the data for Modelling
X = heart.drop(["target"], axis=1)  # Features for classification
Y = heart["target"]  # Target for classification

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Scale features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Support Vector Machine( Support Vector Classifier )
model = svm.SVC(kernel='linear', class_weight='balanced')

model.fit(X_train_scaled, Y_train)

svm_model = model.predict(X_test_scaled)

score_svm = round(accuracy_score(svm_model,Y_test)*100,2)

print(classification_report(Y_test, svm_model))

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")

# serialized model
joblib.dump(svm_model, 'cad_model.pkl')

# Save the scaler too, since you’ll need it in the web app
joblib.dump(scaler, 'scaler.pkl')
