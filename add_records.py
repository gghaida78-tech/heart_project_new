import pandas as pd
import numpy as np

# مسار ملف البيانات
file_path = "C:/Users/Elite/Downloads/heart_comma_updated.csv"


# قراءة الملف الأصلي مع تحديد الفاصل الصحيح
df = pd.read_csv(file_path, sep=';')

print("عدد السجلات الأصلي:", len(df))
print("معاينة البيانات:")
print(df.head())

# تحديد عدد السجلات الجديدة
num_new_records = 10000

# إنشاء بيانات عشوائية جديدة بناءً على نطاقات البيانات الأصلية
new_data = pd.DataFrame({
    'age': np.random.randint(df['age'].min(), df['age'].max()+1, size=num_new_records),
    'sex': np.random.choice(df['sex'].unique(), size=num_new_records),
    'cp': np.random.choice(df['cp'].unique(), size=num_new_records),
    'trtbps': np.random.randint(df['trtbps'].min(), df['trtbps'].max()+1, size=num_new_records),
    'chol': np.random.randint(df['chol'].min(), df['chol'].max()+1, size=num_new_records),
    'fbs': np.random.choice(df['fbs'].unique(), size=num_new_records),
    'restecg': np.random.choice(df['restecg'].unique(), size=num_new_records),
    'thalachh': np.random.randint(df['thalachh'].min(), df['thalachh'].max()+1, size=num_new_records),
    'exng': np.random.choice(df['exng'].unique(), size=num_new_records),
    'oldpeak': np.round(np.random.uniform(df['oldpeak'].min(), df['oldpeak'].max(), size=num_new_records), 1),
    'slp': np.random.choice(df['slp'].unique(), size=num_new_records),
    'caa': np.random.choice(df['caa'].unique(), size=num_new_records),
    'thall': np.random.choice(df['thall'].unique(), size=num_new_records),
    'output': np.random.choice(df['output'].unique(), size=num_new_records),
})

# دمج البيانات الجديدة مع الأصلية
df = pd.concat([df, new_data], ignore_index=True)

# حفظ الملف مرة ثانية مع الفاصل الصحيح
df.to_csv(file_path, sep=';', index=False)

print(f"تمت إضافة {num_new_records} سجل جديد بنجاح!")
print("عدد السجلات الآن:", len(df))
