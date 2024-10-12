import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\ugrkr\OneDrive\Masaüstü\Finals Model.xlsx")

X = df.drop("karzarar_oranı%(günlük)", axis=1)
y = df["karzarar_oranı%(günlük)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

categorical_c = X_train.select_dtypes(include="object").columns.tolist()
numerical_c = X_train.select_dtypes(include=["float", "int"]).columns.tolist()

categorical_pipeline = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

numerical_pipeline = Pipeline(steps=[
    ("scaler", MinMaxScaler()),
    ("imputer", SimpleImputer(strategy="mean"))
])

# Tüm veriler için ön işleme
preprocess = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_c),
    ("cat", categorical_pipeline, categorical_c)
])

# Eğitim setlerini işle
X_train_processed = preprocess.fit_transform(X_train)
X_test_processed = preprocess.transform(X_test)

# LightGBM modelini oluştur ve eğit
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train_processed, y_train)

# Tahminler yap
y_pred = model.predict(X_test_processed)

# Performans metriklerini hesapla
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}, R²: {r2}")

# Sonuçları grafikleştir
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.grid()
plt.show()

# Modeli kaydet
joblib.dump(model, "sehirhatlarılightgbm_model.joblib")
joblib.dump(preprocess, "sehirhatlarıpreprocessor.joblib")
print("Model ve ön işlemci kaydedildi.")
df.columns

df.head()