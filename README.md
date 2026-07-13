# 🎓 Predictive Model for Academic Performance

Proyecto de Inteligencia Artificial cuyo objetivo es predecir la **calificación final (G3)** de estudiantes a partir de variables biográficas, demográficas, familiares y de comportamiento, comparando distintos algoritmos de regresión para identificar el que mejor captura la relación entre hábitos/contexto del estudiante y su rendimiento académico.

## Dataset

- **395 estudiantes**, **33 variables** por estudiante (dataset de rendimiento académico de escuelas secundarias, con variables como sexo, edad, nivel educativo de los padres, tiempo de estudio semanal, consumo de alcohol, tiempo libre, salud, ausencias, etc.).
- Variables objetivo: `G1`, `G2`, `G3` (calificaciones del primer, segundo y tercer período). El modelo final predice **G3**.

## Análisis exploratorio (EDA)

Realizado en [`EDA/EDA.ipynb`](./EDA/EDA.ipynb):

- **Detección de nulos y valores en cero** por columna, y limpieza dirigida (ej. se eliminó la columna `failures` por alta proporción de ceros no informativos).
- **Codificación binaria** de variables categóricas (`school`, `sex`, `address`, `famsize`, `Parrent_status`, apoyo educativo extra, actividades extracurriculares, etc.).
- **Detección y remoción de anomalías/outliers**: estudiantes con edad ≥ 20, `Mother_edu`/`Father_edu` en 0, más de 30 ausencias, y calificaciones (`G1`, `G2`, `G3`) fuera de rangos plausibles.
- **Normalización** de variables numéricas con `StandardScaler`.
- **Clustering exploratorio (K-Means)**: uso del método del codo para determinar el número óptimo de clusters, identificando **4 perfiles distintos de estudiantes** según variables personales, familiares, académicas y de comportamiento.
- **Árbol de decisión exploratorio** para identificar variables más relevantes en la predicción de `G3`, confirmando que `G2` (nota del período anterior) es el predictor más determinante.
- **División de datos**: 80% train+val / 20% test, y del 80% otro 80/20 para train/validación (`train.csv`, `validation.csv`, `test.csv`).

## Modelado — comparación de algoritmos

Se entrenaron y compararon 3 familias de modelos de regresión (carpeta [`Experimentos/`](./Experimentos/)), evaluando distintas combinaciones de hiperparámetros para cada uno:

| Modelo | Mejor R² | MAE | MSE | RMSE |
|---|---|---|---|---|
| Ridge / Lasso (regresión lineal regularizada) | ~0.04 | ~3.77 | ~21.7 | ~4.66 |
| KNN (K-Nearest Neighbors) | ~0.17 | ~3.39 | ~18.8 | ~4.33 |
| **Random Forest** | **~0.30** | **~3.19** | **~15.8** | **~4.00** |

**Random Forest fue el modelo con mejor desempeño** de los tres, aunque ninguno alcanzó una capacidad predictiva alta — lo cual llevó a una conclusión relevante del proyecto: los modelos lineales y basados en distancia (KNN) no logran capturar bien la relación entre estas variables y la nota final, y se identificó como siguiente paso explorar modelos no lineales más complejos (redes neuronales).

## Modelo final y predicción

En [`Prediccion/predicciones.ipynb`](./Prediccion/predicciones.ipynb):

- Se entrena un **Random Forest Regressor** con hiperparámetros fijados a partir de los experimentos (`R²: 0.31 | MAE: 3.22 | MSE: 15.60 | RMSE: 3.95`).
- El modelo y los `LabelEncoder` usados se **serializan con `joblib`** (`random_forest_model.pkl`, `label_encoders.pkl`) para poder reutilizarse sin reentrenar.
- Se implementa una función de **preprocesamiento de datos nuevos** (`preprocess_new_data`) que aplica los mismos encoders guardados, permitiendo:
  - Predecir la nota `G3` de **todo un conjunto de estudiantes nuevos** (`test.csv`).
  - Predecir la nota de **un único estudiante** dado su índice.

## Tecnologías utilizadas

| Categoría | Herramienta |
|---|---|
| Lenguaje | Python |
| Manipulación de datos | pandas, numpy |
| Modelado | scikit-learn (Ridge, Lasso, Random Forest, KNN, K-Means, Decision Tree) |
| Visualización | matplotlib, seaborn |
| Persistencia de modelos | joblib |
| Entorno | Jupyter Notebook |

## 📁 Estructura del repositorio

```
Predictive-Model-for-Academic-Performance/
├── Data/
│   ├── students.csv          # Dataset original
│   ├── train.csv              # 64% de los datos (entrenamiento)
│   ├── validation.csv          # 16% de los datos (validación de hiperparámetros)
│   └── test.csv                 # 20% de los datos (evaluación final)
├── EDA/
│   └── EDA.ipynb                 # Limpieza, análisis exploratorio y clustering
├── Experimentos/
│   ├── Regresion_Logistica_Ridge_Lasso.ipynb
│   ├── Random_Forest.ipynb
│   ├── KNN.ipynb
│   └── conclusion.txt             # Conclusiones comparativas de los 3 modelos
├── Prediccion/
│   └── predicciones.ipynb          # Entrenamiento final, serialización y predicción
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

### Prerrequisitos
- Python 3.10+
- Jupyter Notebook / JupyterLab

### Pasos

```bash
git clone https://github.com/alejandrocanadas/Predictive-Model-for-Academic-Performance-Artificial-Intelligence-Project.git
cd Predictive-Model-for-Academic-Performance-Artificial-Intelligence-Project
pip install -r requirements.txt
jupyter notebook
```

Orden recomendado para reproducir el flujo completo:
1. `EDA/EDA.ipynb` — limpieza de datos y generación de `train.csv` / `validation.csv` / `test.csv`.
2. `Experimentos/*.ipynb` — comparación de modelos e hiperparámetros.
3. `Prediccion/predicciones.ipynb` — entrenamiento del modelo final y generación de predicciones.

## 📌 Conclusiones clave

- El desempeño académico previo (`G2`) es, por lejos, la variable más predictiva de la nota final.
- De los modelos evaluados, **Random Forest** ofrece el mejor balance entre error y capacidad de generalización, aunque el poder predictivo global sigue siendo limitado (R² ≈ 0.30).
- Los modelos lineales (Ridge, Lasso) y basados en distancia (KNN) no son adecuados para este problema con las variables disponibles, sugiriendo como trabajo futuro el uso de modelos no lineales más complejos (redes neuronales).

