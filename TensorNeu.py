import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    if problem_type == "regression":
        scaler_Y = StandardScaler()
        Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
    else:
        Y_scaled = Y

    return X_scaled, Y_scaled, problem_type, scaler_X  


X, Y, problem_type, scaler_X = prepare_data("3_211110 (1).csv")


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],), activation="sigmoid" if problem_type == "classification" else "linear")
])


loss_function = "binary_crossentropy" if problem_type == "classification" else "mse"
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss_function, metrics=['accuracy' if problem_type == "classification" else 'mse'])


initial_weights, initial_bias = model.layers[0].get_weights()


history = model.fit(X, Y, epochs=70, batch_size=32, verbose=1)


final_weights, final_bias = model.layers[0].get_weights()


final_error = history.history['loss'][-1]


num_epochs = len(history.history['loss'])


print("\n Reporte de los Pesos ")
print(f"W₀ (Pesos Iniciales): {initial_weights.flatten()}")
print(f"Wₓ (Pesos Finales): {final_weights.flatten()}")
print(f"Bias Inicial: {initial_bias}")
print(f"Bias Final: {final_bias}")
print(f"Error final |error|: {final_error}")
print(f"Número total de épocas: {num_epochs}")


results = model.evaluate(X, Y)
print("\nEvaluación del modelo:", results)


predictions = model.predict(X)


weights, bias = model.layers[0].get_weights()
print("\nPesos finales:", weights.flatten())
print("Bias final:", bias)


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label="Error")
plt.xlabel("Época")
plt.ylabel("Error")
plt.title("Evolución del Error durante el Entrenamiento")
plt.legend()
plt.grid(True)
plt.show()


Y_pred = model.predict(X)
sample_errors = np.abs(Y_pred.flatten() - Y)

plt.figure(figsize=(12, 6))
plt.scatter(range(len(sample_errors)), sample_errors, color='red', marker='o', label="|Y_c - Y_d| (Error absoluto)")
plt.xlabel("Índice de la muestra")
plt.ylabel("Error absoluto")
plt.title("Error absoluto entre Valores Predichos y Reales")
plt.legend()
plt.grid(True)
plt.show()