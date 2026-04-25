## DATASET YANG DIGUNAKAN MERUPAKAN HASIL MERGE (PENGGABUNGAN) DARI DATASET 'Malnutrition.csv' DAN 'Stunting.csv' Kaggle
## INI DILAKUKAN AGAR MEMUDAHKAN PENGEMBANGAN APLIKASi DENGAN MENGGUNAKAN DATASET YANG LENGKAP DAN MENDUKUNG

import pandas as pd

df_a = pd.read_csv("malnutrition.csv")
df_b = pd.read_csv("stunting.csv")

df_a = df_a.rename(columns={
    'age': 'Age',
    'height': 'Height',
    'weight': 'Weight',
    'bmi': 'BMI',
    'muac': 'MUAC'
})

df_b = df_b.rename(columns={
    'umur': 'Age',
    'tinggi_badan': 'Height',
})

# contoh: cm → meter
df_a['Height'] = df_a['Height'] / 100
df_b['Height'] = df_b['Height'] / 100

df_b['BMI'] = df_b['Weight'] / (df_b['Height'] ** 2)

df_b['MUAC'] = None
df_a['Head_Circumference'] = None
df_b['Head_Circumference'] = None

df_combined = pd.concat([df_a, df_b], ignore_index=True)

# hapus duplikat
df_combined = df_combined.drop_duplicates()

# handle missing values
df_combined = df_combined.fillna(df_combined.median(numeric_only=True))

df_combined['Stunting'] = (df_combined['Height'] < threshold).astype(int)

df_combined.to_csv("clean_stunting_dataset.csv", index=False)
