import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv("train.csv")
test_kaggle = pd.read_csv("test.csv")

train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)

# -------- Preprocessing Function -------- #
def preprocess(df, is_train=True):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    if is_train:
        X = df.drop(['Survived', 'PassengerId'], axis=1)
        y = df['Survived']
        return X, y
    else:
        passenger_ids = df['PassengerId']
        X = df.drop(['PassengerId'], axis=1)
        return X, passenger_ids

# Preprocess training/testing for model
X_train, y_train = preprocess(train_data)
X_test, y_test = preprocess(test_data)



# Preprocess Kaggle test.csv
X_kaggle_test, kaggle_passenger_ids = preprocess(test_kaggle, is_train=False)











# -------- Train Random Forest -------- #
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# -------- Train Logistic Regression -------- #
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# -------- Predictions -------- #
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)

# -------- Confusion Matrix -------- #
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, y_pred_rf))
print("\nLogistic Regression Confusion Matrix")
print(confusion_matrix(y_test, y_pred_lr))

# -------- Feature Coefficients Logistic Regression -------- #
coefs = pd.Series(lr.coef_[0], index=X_train.columns)
plt.figure(figsize=(8, 5))
coefs.sort_values().plot(kind='barh')
plt.title("Logistic Regression Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.show()

# -------- Visualize Tree from Random Forest -------- #
plt.figure(figsize=(20, 10))
plot_tree(rf.estimators_[0], feature_names=X_train.columns, class_names=['Not Survived', 'Survived'], filled=True, max_depth=3)
plt.title("Random Forest (Tree Example)")
plt.show()

# -------- Create Kaggle Submission Files -------- #
# Predict on Kaggle test set (418 rows)
kaggle_pred_rf = rf.predict(X_kaggle_test)
kaggle_pred_lr = lr.predict(X_kaggle_test)

# Save submissions
submission_rf = pd.DataFrame({'PassengerId': kaggle_passenger_ids, 'Survived': kaggle_pred_rf})
submission_rf.to_csv('submission_random_forest.csv', index=False)

submission_lr = pd.DataFrame({'PassengerId': kaggle_passenger_ids, 'Survived': kaggle_pred_lr})
submission_lr.to_csv('submission_logistic_regression.csv', index=False)

print("Submission files created successfully.")