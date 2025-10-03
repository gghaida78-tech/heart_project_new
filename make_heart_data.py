import pandas as pd
import numpy as np

# عدد السجلات الجديدة
num_records = 10000

# توليد بيانات عشوائية قريبة من بيانات أمراض القلب
data = {
    "age": np.random.randint(29, 77, num_records),          # العمر
    "sex": np.random.randint(0, 2, num_records),            # الجنس (0 = أنثى, 1 = ذكر)
    "cp": np.random.randint(0, 4, num_records),             # نوع ألم الصدر
    "trtbps": np.random.randint(90, 200, num_records),      # ضغط الدم
    "chol": np.random.randint(120, 570, num_records),       # الكولسترول
    "fbs": np.random.randint(0, 2, num_records),            # سكر صائم > 120
    "restecg": np.random.randint(0, 2, num_records),        # تخطيط القلب
    "thalachh": np.random.randint(70, 210, num_records),    # أعلى معدل نبض
    "exng": np.random.randint(0, 2, num_records),           # ذبحة عند المجهود
    "oldpeak": np.round(np.random.uniform(0.0, 6.2, num_records), 1),  # انخفاض ST
    "slp": np.random.randint(0, 3, num_records),            # ميل القطع ST
    "caa": np.random.randint(0, 5, num_records),            # عدد الأوعية
    "thall": np.random.randint(0, 4, num_records),          # نوع thal
    "output": np.random.randint(0, 2, num_records)          # النتيجة (0 أو 1)
}

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data)

# حفظ البيانات في ملف CSV جديد
file_path = r"C:/Users/Elite/Downloads/heart_comma_updated.csv"
df.to_csv(file_path, sep=';', index=False)

print(f"✅ تم إنشاء الملف وحفظ {num_records} سجل في {file_path}")
