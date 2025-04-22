# ridge_lasso_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso
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

# Función para generar y evaluar los modelos Ridge y Lasso
def evaluate_ridge_lasso(X_train, X_test, y_train, y_test, model_type='ridge'):
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)  # Alpha controla la regularización L2
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)  # Alpha controla la regularización L1
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    
    return r2, mae, mse, rmse

# Anidar tres ciclos `for` para probar combinaciones de tres variables
for var1 in variables:
    for var2 in variables:
        for var3 in variables:
            if var1 != var2 and var1 != var3 and var2 != var3:  # Evitar combinaciones repetidas
                # Seleccionar las características basadas en la combinación de variables
                X_subset = X[[var1, var2, var3]]

                # Evaluar Ridge
                r2_ridge, mae_ridge, mse_ridge, rmse_ridge = evaluate_ridge_lasso(X_train, X_test, y_train, y_test, model_type='ridge')
                print(f"Combinación de variables (Ridge): {var1}, {var2}, {var3}")
                print(f"R2: {r2_ridge}, MAE: {mae_ridge}, MSE: {mse_ridge}, RMSE: {rmse_ridge}")
                print("-" * 50)

                # Evaluar Lasso
                r2_lasso, mae_lasso, mse_lasso, rmse_lasso = evaluate_ridge_lasso(X_train, X_test, y_train, y_test, model_type='lasso')
                print(f"Combinación de variables (Lasso): {var1}, {var2}, {var3}")
                print(f"R2: {r2_lasso}, MAE: {mae_lasso}, MSE: {mse_lasso}, RMSE: {rmse_lasso}")
                print("=" * 50)
