import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ===============================
# 1. Carga y Preprocesamiento del Dataset
# ===============================

# Ruta del dataset (ajustar si es necesario)
DATASET_PATH = "dataset"

# Función para validar imágenes
import cv2

def validar_imagenes(dataset_path):
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error: No se pudo cargar la imagen {img_path}")
                elif img.shape != (24, 24):
                    print(f"Advertencia: Imagen {img_path} no tiene tamaño 24x24, tiene {img.shape}")

# Validar imágenes
validar_imagenes(DATASET_PATH)

# Cargamos el dataset
train_ds = image_dataset_from_directory(
    DATASET_PATH,
    image_size=(24, 24),
    color_mode='grayscale',  # Para imágenes en blanco y negro
    batch_size=32,
    seed=42
)

# Obtener nombres de clases
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Clases detectadas: {class_names}")

# Normalizamos las imágenes
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# ===============================
# 2. Definición y Entrenamiento del Modelo
# ===============================

# Definir el modelo sin capa oculta
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24, 24, 1)),  # Convertir imagen en vector
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # Capa oculta
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_ds, epochs=70)

# ===============================
# 3. Validación Cruzada
# ===============================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X, y = [], []
for images, labels in train_ds:
    X.extend(images.numpy())
    y.extend(labels.numpy())
X, y = np.array(X), np.array(y)

val_errors = []
train_errors = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model.fit(X_train, y_train, epochs=10, verbose=0)
    train_loss, _ = model.evaluate(X_train, y_train, verbose=0)
    val_loss, _ = model.evaluate(X_val, y_val, verbose=0)
    
    train_errors.append(train_loss)
    val_errors.append(val_loss)

# ===============================
# 4. Cálculo de Errores
# ===============================

error_entrenamiento = np.mean(train_errors)
error_validacion = np.mean(val_errors)
error_total = error_entrenamiento + error_validacion
print(f"Error de Entrenamiento: {error_entrenamiento}")
print(f"Error de Validación: {error_validacion}")
print(f"Error Total: {error_total}")

# ===============================
# 5. Matriz de Confusión
# ===============================

y_pred = np.argmax(model.predict(X), axis=1)
conf_matrix = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.title("Matriz de Confusión")
plt.show()

# ===============================
# 6. Resumen de Matriz de Confusión
# ===============================

print("\nReporte de Clasificación:")
print(classification_report(y, y_pred, target_names=class_names))
