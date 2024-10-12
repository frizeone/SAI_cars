import numpy as np
import matplotlib.pyplot as plt

# Функция для создания единичной матрицы с использованием NumPy
def create_identity_matrix_numpy(n):
    return np.eye(n)

# Визуализация данных
def visualize_data(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Зависимость прибыли от количества автомобилей')
    plt.show()

# Функция для вычисления стоимости с использованием NumPy
def compute_cost_numpy(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

# Градиентный спуск с использованием NumPy
def gradient_descent_numpy(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)
        J_history.append(compute_cost_numpy(X, y, theta))
    return theta, J_history

# Функция для предсказания с использованием NumPy
def predict_numpy(theta, X_new):
    X_new = np.array([1, X_new])  # Добавляем 1 для theta_0
    return X_new.dot(theta)

# Функция для записи результатов в файл
def write_to_file(results, filename="output.txt"):
    with open(filename, 'w') as f:
        f.write(results)
