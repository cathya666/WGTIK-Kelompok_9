import pandas as pd
import numpy as np

#1. LOAD DATA

url = "https://raw.githubusercontent.com/cathya666/WGTIK-Kelompok_9/4d1eb9c0a9ed1ebc9c3c717dc3211e48b657b91c/DATASETS/malnutrition_data%20(1).csv"
df = pd.read_csv(url)

print("dataset berhasil dimuat")
print("shape:", df.shape)

#2. DATA SEBELUM PREPROCESSING
print("\ninfo data:")
print(df.info())

print("\nstatistik deskriptif:")
print(df.describe())

print("\n5 baris pertama:")
df.head()

#3. CLEANING DATA
#missing value
print("cek missing value:")
print(df.isnull().sum())

# kolom bmi hampir semua isinya 10.0, bukan nilai asli
# replace ke NaN dulu baru isi pakai median
print("\njumlah bmi == 10.0:", (df['bmi'] == 10.0).sum())

df['bmi'] = df['bmi'].replace(10.0, np.nan)
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

print("missing value setelah ditangani:")
print(df.isnull().sum())

#duplikasi
print("\njumlah duplikasi:", df.duplicated().sum())
df = df.drop_duplicates()
print("jumlah duplikasi setelah dihapus:", df.duplicated().sum())

#4. FORMATTING
#rename kolom
df = df.rename(columns={
    'age_months'      : 'Age_Months',
    'weight_kg'       : 'Weight_kg',
    'height_cm'       : 'Height_cm',
    'muac_cm'         : 'MUAC_cm',
    'bmi'             : 'BMI',
    'nutrition_status': 'Nutrition_Status'
})

print("kolom setelah rename:")
print(df.columns.tolist())

#konversi tipe data
df['Age_Months'] = df['Age_Months'].round(0).astype(int)
df['Nutrition_Status'] = df['Nutrition_Status'].astype('category')

print("\ntipe data setelah konversi:")
print(df.dtypes)

#5. FILTERING & TRANSFORMASI
#filter usia 0-60 bulan (sesuai definisi balita)
df = df[df['Age_Months'] <= 60].reset_index(drop=True)
print("shape setelah filter usia:", df.shape)

#hitung ulang BMI pakai rumus berat(kg) / tinggi(m)^2
df['BMI'] = (df['Weight_kg'] / ((df['Height_cm'] / 100) ** 2)).round(2)

print("\nBMI setelah dihitung ulang:")
print(df['BMI'].describe())

#DATA SETELAH PREPROCESSING
print("\nshape akhir:", df.shape)
print("\n10 baris pertama setelah preprocessing:")
df.head(10)
