## DATASET YANG DIGUNAKAN MERUPAKAN HASIL MERGE (PENGGABUNGAN) DARI DATASET 'Malnutrition.csv' DAN 'Stunting.csv' Kaggle
## INI DILAKUKAN AGAR MEMUDAHKAN PENGEMBANGAN APLIKASi DENGAN MENGGUNAKAN DATASET YANG LENGKAP DAN MENDUKUNG

import pandas as pd
import warnings

# Load datasets
url_a = "https://raw.githubusercontent.com/cathya666/WGTIK-Kelompok_9/4d1eb9c0a9ed1ebc9c3c717dc3211e48b657b91c/DATASETS/malnutrition_data%20(1).csv"
url_b = "https://raw.githubusercontent.com/cathya666/WGTIK-Kelompok_9/4d1eb9c0a9ed1ebc9c3c717dc3211e48b657b91c/DATASETS/data_balita.csv"

df_a = pd.read_csv(url_a)
df_b = pd.read_csv(url_b)

# Rename columns
df_a = df_a.rename(columns={
    'age_months': 'Age',
    'weight_kg': 'Weight',
    'height_cm': 'Height',
    'muac_cm': 'MUAC'
})

df_b = df_b.rename(columns={
    'Umur (bulan)': 'Age',
    'Tinggi Badan (cm)': 'Height',
    'Jenis Kelamin': 'Gender',
    'Status Gizi': 'Nutrition_Status'
})

# Add missing columns
df_a['BMI'] = None
df_a['Head_Circumference'] = None
df_a['Gender'] = None
df_a['Nutrition_Status'] = df_a.get('nutrition_status', None)

df_b['BMI'] = None
df_b['Weight'] = None
df_b['MUAC'] = None
df_b['Head_Circumference'] = None

# Ensure columns exist
required_cols = ['Age', 'Height', 'Weight', 'BMI', 'MUAC', 
                 'Head_Circumference', 'Gender', 'Nutrition_Status']

for col in required_cols:
    if col not in df_a.columns:
        df_a[col] = None
    if col not in df_b.columns:
        df_b[col] = None

# **FIX: Hanya gabungkan kolom yang diperlukan**
df_a = df_a[required_cols]
df_b = df_b[required_cols]

# **FIX: Ubah tipe data untuk menghindari warning**
for col in required_cols:
    if df_a[col].isna().all() and df_b[col].isna().all():
        df_a[col] = df_a[col].astype('float64')
        df_b[col] = df_b[col].astype('float64')

# Concatenate tanpa warning
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

# Remove duplicates
df_combined = df_combined.drop_duplicates()

# Fill missing values
for col in ['Age', 'Height']:
    if col in df_combined.columns and df_combined[col].notna().any():
        median_value = df_combined[col].median()
        df_combined[col] = df_combined[col].fillna(median_value)

# Calculate stunting
threshold = 0.85
if 'Height' in df_combined.columns and df_combined['Height'].notna().any():
    df_combined['Stunting'] = (df_combined['Height'] < threshold).astype(int)
else:
    df_combined['Stunting'] = None

# Save to CSV
df_combined.to_csv("clean_stunting_dataset.csv", index=False)

print(f"Dataset successfully created with shape: {df_combined.shape}")
print(f"Total rows: {len(df_combined)}")
print(f"Columns: {list(df_combined.columns)}")
