import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# 1. Cargar los datos
file_path = "211110.data"
df = pd.read_csv(file_path, header=None, na_values="?", low_memory=False)

# Renombrar las columnas según la documentación
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# 2. Aplicar la nueva estrategia de limpieza de datos

# Contar valores nulos por columna
missing_values = df.isnull().sum()

# Definir umbral del 50% para eliminar columnas
threshold = len(df) * 0.5

# Identificar columnas a eliminar (más del 50% de valores nulos)
columns_to_drop = missing_values[missing_values > threshold].index.tolist()

# Eliminar columnas con demasiados datos faltantes
df_cleaned = df.drop(columns=columns_to_drop)

# Eliminar filas con valores faltantes en las columnas restantes
df_cleaned = df_cleaned.dropna()

# 3. Separar características (X) y variable objetivo (y)
X = df_cleaned.drop(columns=["num"]).values
y = df_cleaned["num"].values

# Mezclar los datos antes de hacer K-Fold Cross Validation
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# Definir el número máximo de clases en la variable objetivo
num_classes = len(np.unique(y))
class_labels = np.sort(np.unique(y))

# 4. Aplicar K-Fold Cross Validation con shuffle habilitado
kf = KFold(n_splits=5, shuffle=True, random_state=42)
history_list = []
conf_matrix_total = np.zeros((num_classes, num_classes))  # Matriz de confusión acumulativa

# Configurar gráfico de pérdida
plt.figure(figsize=(10, 6))

# Definir tasa de aprendizaje
learning_rate = 0.001

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 5. Definir y entrenar la red neuronal con 1 neurona
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)), 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, validation_data=(X_test, y_test))
    history_list.append(history)

    # Graficar evolución del error
    plt.plot(history.history['loss'], label=f'Fold {fold+1}')

    # Predicciones y matriz de confusión individual
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)

    # Acumular la matriz de confusión total
    conf_matrix_total += conf_matrix

# Configurar gráfica de error
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Loss)')
plt.title('Evolución del Error en cada Fold')
plt.legend()
plt.show()

# 6. Mostrar Matriz de Confusión Final
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_total.astype(int), annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión Total Acumulada')
plt.show()
