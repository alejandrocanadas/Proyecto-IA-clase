# knn_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Cargar el dataset
data = pd.read_csv('students.csv')

# Selección de características (X) y la etiqueta (y)
X = data.drop(['G1', 'G2', 'G3'], axis=1)  # Eliminar las columnas G1, G2
y = data['G3']  # Usamos la calificación final (G3) como objetivo

# Preprocesar las variables categóricas
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista de variables para probar
variables = X.columns.tolist()

# Función para generar y evaluar el modelo k-NN
def evaluate_knn(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    
    r2_knn = r2_score(y_test, knn_pred)
    mae_knn = mean_absolute_error(y_test, knn_pred)
    mse_knn = mean_squared_error(y_test, knn_pred)
    rmse_knn = np.sqrt(mse_knn)
    
    return r2_knn, mae_knn, mse_knn, rmse_knn

# Anidar tres ciclos `for` para probar combinaciones de tres variables
for var1 in variables:
    for var2 in variables:
        for var3 in variables:
            if var1 != var2 and var1 != var3 and var2 != var3:  # Evitar combinaciones repetidas
                # Seleccionar las características basadas en la combinación de variables
                X_subset = X[[var1, var2, var3]]

                # Evaluar k-NN
                r2_knn, mae_knn, mse_knn, rmse_knn = evaluate_knn(X_train, X_test, y_train, y_test)
                print(f"Combinación de variables (k-NN): {var1}, {var2}, {var3}")
                print(f"R2: {r2_knn}, MAE: {mae_knn}, MSE: {mse_knn}, RMSE: {rmse_knn}")
                print("-" * 50)
