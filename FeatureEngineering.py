# ======== Import Libraries ========
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import warnings

warnings.filterwarnings('ignore')

# ======== Load Dataset ========
train_df = pd.read_csv('train.csv')
test_kaggle = pd.read_csv('test.csv')

# ======== Preprocessing Function ========
def preprocess(df, is_train=True):
    df = df.copy()

    # Fill missing
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(df['Title'].value_counts()
                                      [df['Title'].value_counts() < 10].index, 'Rare')

    for col in ['Sex', 'Embarked', 'Title']:
        df[col] = LabelEncoder().fit_transform(df[col])

    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    if is_train:
        X = df.drop(['Survived', 'PassengerId'], axis=1)
        y = df['Survived']
        return X, y
    else:
        passenger_ids = df['PassengerId']
        X = df.drop(['PassengerId'], axis=1)
        return X, passenger_ids

# ======== Split for Train/Test (80/20) ========
train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)
X_train, y_train = preprocess(train_data, is_train=True)
X_test, y_test = preprocess(test_data, is_train=True)

# ======== Preprocess Kaggle Test Set ========
X_kaggle_test, kaggle_passenger_ids = preprocess(test_kaggle, is_train=False)

# ======== Scaling (for Logistic Regression) ========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_kaggle_scaled = scaler.transform(X_kaggle_test)

# ======== Train Models ========
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# ======== Predictions ========
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test_scaled)

# ======== Confusion Matrix ========
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))
print("\nLogistic Regression Confusion Matrix")
print(confusion_matrix(y_test, y_pred_lr))

# ======== Logistic Regression Coefficients ========
coefs = pd.Series(lr.coef_[0], index=X_train.columns)
plt.figure(figsize=(8, 5))
coefs.sort_values().plot(kind='barh')
plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()

# ======== Random Forest Tree Visualization ========
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], 
          feature_names=X_train.columns, 
          class_names=["Not Survived", "Survived"],
          filled=True,
          max_depth=3)
plt.title("Random Forest Tree Example")
plt.show()

# ======== Create Submission Files ========
kaggle_pred_rf = rf.predict(X_kaggle_test)
kaggle_pred_lr = lr.predict(X_kaggle_scaled)

submission_rf = pd.DataFrame({
    'PassengerId': kaggle_passenger_ids,
    'Survived': kaggle_pred_rf
})
submission_rf.to_csv('feature_engineering_submission_random_forest.csv', index=False)

submission_lr = pd.DataFrame({
    'PassengerId': kaggle_passenger_ids,
    'Survived': kaggle_pred_lr
})
submission_lr.to_csv('feature_engineering_submission_logistic_regression.csv', index=False)

# ======== Confusion Matrix (Visual) ========
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Random Forest Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf,
    display_labels=["Not Survived", "Survived"],
    cmap=plt.cm.Blues
)
plt.title("Random Forest Confusion Matrix")
plt.grid(False)
plt.show()

# Logistic Regression Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr,
    display_labels=["Not Survived", "Survived"],
    cmap=plt.cm.Greens
)
plt.title("Logistic Regression Confusion Matrix")
plt.grid(False)
plt.show()

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)

# ======== Evaluation Metrics for Random Forest ========
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("🎯 Random Forest Metrics")
print(f"Accuracy : {acc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall   : {recall_rf:.4f}")
print(f"F1-score : {f1_rf:.4f}")
print()

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print("🎯 Logistic Regression")
print(f"Accuracy : {acc_lr:.4f}")
print(f"Precision: {prec_lr:.4f}")
print(f"Recall   : {recall_lr:.4f}")
print(f"F1-score : {f1_lr:.4f}")
print()


print("✅ Submission files created successfully.")

