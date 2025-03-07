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
    return X_scaled, Y, problem_type, scaler_X

X, Y, problem_type, scaler_X = prepare_data("3_211110 (1).csv")

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, Y = X[indices], Y[indices]

k = 5
fold_size = len(X) // k
all_errors = []

for i in range(k):
    start = i * fold_size
    end = start + fold_size if i < k - 1 else len(X)
    X_val, Y_val = X[start:end], Y[start:end]
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    Y_train = np.concatenate((Y[:start], Y[end:]), axis=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X.shape[1],), activation="sigmoid" if problem_type == "classification" else "linear")
    ])

    loss_function = "binary_crossentropy" if problem_type == "classification" else "mse"
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=loss_function, metrics=['mse'])

    initial_weights, initial_bias = model.layers[0].get_weights()
    history = model.fit(X_train, Y_train, epochs=70, batch_size=32, verbose=0, validation_data=(X_val, Y_val))

    final_weights, final_bias = model.layers[0].get_weights()
    train_error = history.history['loss'][-1]
    val_error = history.history['val_loss'][-1]
    all_errors.append(history.history['loss'])

    Y_pred = model.predict(X_val)
    abs_error_total = np.sum(np.abs(Y_val - Y_pred.flatten()))

    print(f"Fold {i+1}:")
    print(f"Pesos iniciales: {initial_weights.flatten()}")
    print(f"Pesos finales: {final_weights.flatten()}")
    print(f"Bias inicial: {initial_bias}")
    print(f"Bias final: {final_bias}")
    print(f"Error de entrenamiento: {train_error:.6f}")
    print(f"Error de validación: {val_error:.6f}")
    print(f"Error absoluto total: {abs_error_total:.6f}")
    print(f"Número de épocas: {len(history.history['loss'])}\\n")

plt.figure(figsize=(10, 5))
for i, errors in enumerate(all_errors):
    plt.plot(errors, label=f'Fold {i+1}')
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.title("Evolución del Error en K-Fold Cross-Validation con Shuffle")
plt.legend()
plt.grid(True)
plt.show()
