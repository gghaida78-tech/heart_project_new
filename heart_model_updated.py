#%% بيانات واستيراد المكتبات
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import sys

#%% قراءة البيانات الجديدة
data = pd.read_csv("C:/Users/Elite/Downloads/heart_comma_updated.csv", sep=';')
print("أول 5 صفوف من البيانات:")
print(data.head())

#%% معلومات عن البيانات
print(data.info())

#%% عدد القيم الفارغة في كل عمود
print(data.isnull().sum())

#%% إحصاءات وصفية للأعمدة الرقمية
print(data.describe())

#%% خريطة الحرارة للمتغيرات
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='Reds')
plt.title("خريطة الحرارة للمتغيرات")
plt.show()

#%% تقسيم البيانات إلى خصائص وهدف
X = data.drop('output', axis=1)
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("حجم بيانات التدريب:", X_train.shape)
print("حجم بيانات الاختبار:", X_test.shape)

#%% تدريب نموذج عشوائي الغابة
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#%% تقييم النموذج
y_pred = model.predict(X_test)
print("\nدقة النموذج على مجموعة الاختبار:", accuracy_score(y_test, y_pred))
print("\nمصفوفة الالتباس:")
print(confusion_matrix(y_test, y_pred))
print("\nالتقرير التفصيلي:")
print(classification_report(y_test, y_pred))

#%% حفظ النموذج المدرب
joblib.dump(model, "C:/Users/Elite/Downloads/heart_model_updated.pkl")
print("✅ تم حفظ النموذج بنجاح في Downloads!")

#%% عرض مسار البايثون الحالي
print("مسار بايثون الحالي:", sys.executable)
