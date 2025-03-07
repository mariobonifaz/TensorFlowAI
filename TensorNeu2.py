import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

def detect_problem_type(Y):
    unique_values = np.unique(Y)
    if np.array_equal(unique_values, [0, 1]):
        return "classification"
    else:
        return "regression"

def detect_delimiter(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        if ";" in first_line:
            return ";"
        elif "," in first_line:
            return ","
        else:
            return ","

def load_dataset(filename):
    delimiter = detect_delimiter(filename)
    df = pd.read_csv(filename, delimiter=delimiter)
    if not isinstance(df.iloc[0, 0], (int, float)):
        df = df.iloc[1:]
    df = df.astype(float)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    return X, Y

def prepare_data(filename):
    X, Y = load_dataset(filename)
    problem_type = detect_problem_type(Y)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    return X_scaled, Y, problem_type, scaler_X

X, Y, problem_type, scaler_X = prepare_data("3_211110 (1).csv")

learning_rates = []
all_errors = []

for i in range(10):
    lr = random.uniform(0.1, 3)
    learning_rates.append(lr)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X.shape[1],), activation="sigmoid" if problem_type == "classification" else "linear")
    ])

    loss_function = "binary_crossentropy" if problem_type == "classification" else "mse"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss_function, metrics=['accuracy' if problem_type == "classification" else 'mse'])

    initial_weights, initial_bias = model.layers[0].get_weights()

    history = model.fit(X, Y, epochs=70, batch_size=32, verbose=0)

    final_weights, final_bias = model.layers[0].get_weights()
    final_error = history.history['loss'][-1]
    all_errors.append(history.history['loss'])

    print(f"Repetición {i+1}: Tasa de aprendizaje={lr:.4f}")
    print(f"Pesos iniciales: {initial_weights.flatten()}")
    print(f"Pesos finales: {final_weights.flatten()}")
    print(f"Bias inicial: {initial_bias}")
    print(f"Bias final: {final_bias}")
    print(f"Error final={final_error:.6f}\\n")

plt.figure(figsize=(10, 5))
for i, errors in enumerate(all_errors):
    plt.plot(errors, label=f'Rep {i+1} (lr={learning_rates[i]:.2f})')
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Evolución del Error para Diferentes Tasas de Aprendizaje")
plt.legend()
plt.grid(True)
plt.show()
