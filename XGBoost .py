# 1. Import ไลบรารีที่จำเป็น
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

# 2. โหลดข้อมูล
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = pd.concat([train, test], sort=False)  # รวม train และ test เพื่อทำ Feature Engineering พร้อมกัน

# 3. Feature Engineering

# 3.1 ดึงคำนำหน้าชื่อ (Title) จาก Name
data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# 3.2 แปลง Sex เป็นตัวเลข
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

# 3.3 เติม Missing Values
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())

# 3.4 FareBin และ AgeBin (แบ่งเป็นกลุ่ม)
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)
data['AgeBin'] = pd.cut(data['Age'].astype(int), 5, labels=False)

# 3.5 FamilySize และ IsAlone
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)


# 3.6 One-hot encoding ให้กับ Embarked และ Title
data = pd.get_dummies(data, columns=['Embarked', 'Title'], drop_first=True)

data.to_csv('prepare_data.csv', index=False)
print("📄 prepare_data.csv created.")


# 4. แยกกลับเป็น train/test อีกครั้ง
train = data[:len(train)]
test = data[len(train):]

# 5. เลือกฟีเจอร์ที่จะใช้กับโมเดล
feature_cols = ['Sex', 'FareBin', 'AgeBin', 'FamilySize', 'IsAlone'] + \
               [col for col in train.columns if col.startswith('Embarked_') or col.startswith('Title_')]

X = train[feature_cols]
y = train['Survived']
X_test = test[feature_cols]

# 6. แบ่ง Training/Validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. สร้างโมเดล + ใช้ GridSearchCV หา hyperparameter ที่ดีที่สุด
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1]
}

grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring=make_scorer(f1_score),
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# 8. ใช้โมเดลที่ดีที่สุดทำนาย
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
f1 = f1_score(y_val, y_pred)
print("Best Parameters:", grid_search.best_params_)
print(f"Validation F1 Score: {f1:.4f}")

# 9. ทำนายบนชุด test
pred_test = best_model.predict(X_test)

# 10. สร้างไฟล์ submission.csv สำหรับ Kaggle
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred_test
})
submission.to_csv('xgboost_submission.csv', index=False)
print("xgboost_submission.csv created.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# คำนวณ metrics ต่าง ๆ
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

# แสดงผลลัพธ์
print("Evaluation Metrics on Validation Set:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# แสดง Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 12. แสดง Feature Importance
from xgboost import plot_importance, plot_tree

plot_importance(best_model, importance_type='gain', max_num_features=10, height=0.5)
plt.title("Top 10 Important Features (by Gain)")
plt.show()

from xgboost import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_model, tree_idx=0, rankdir='LR')
plt.title("XGBoost Tree Visualization (Tree 0)", fontsize=16)
plt.show()
