# model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

# Definir la función para generar el modelo
def generateModel(X_train, X_test, y_train, y_test):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Salida continua para predecir calificación
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Evaluación del modelo
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return r2, mae, mse, rmse


# Función principal para entrenar con combinaciones de variables
def train_model_with_variable_combinations():
    # Cargar el dataset
    data = pd.read_csv('students.csv')

    # Selección de características (X) y la etiqueta (y)
    X = data.drop(['G1', 'G2', 'G3'], axis=1)
    y = data['G3']  # Usamos la calificación final (G3) como objetivo

    # Preprocesar las variables categóricas
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))
        label_encoders[column] = le

    # Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Variables que deseas probar (puedes modificar esto según tus necesidades)
    variables = ['age', 'sex', 'school']  # Ejemplo con algunas variables

    # Anidar tres ciclos for para probar combinaciones de tres variables
    for var1 in variables:
        for var2 in variables:
            for var3 in variables:
                if var1 != var2 and var1 != var3 and var2 != var3:  # Para evitar que se repitan variables
                    # Seleccionar las características basadas en las combinaciones de variables
                    X_subset = X[[var1, var2, var3]]

                    # Generar el modelo y calcular las métricas
                    r2, mae, mse, rmse = generateModel(X_train, X_test, y_train, y_test)

                    # Mostrar los resultados
                    print(f"Combinación de variables: {var1}, {var2}, {var3}")
                    print(f"R2 Score: {r2}")
                    print(f"MAE: {mae}")
                    print(f"MSE: {mse}")
                    print(f"RMSE: {rmse}")
                    print("-" * 50)

if __name__ == "__main__":
    train_model_with_variable_combinations()
