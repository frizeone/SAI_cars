import numpy as np
import matplotlib.pyplot as plt

# 1. Функция для создания единичной матрицы с использованием NumPy
def create_identity_matrix_numpy(n):

    # Создаёт единичную матрицу размером n x n с использованием NumPy.

    return np.eye(n)

# 2. Самописная функция для создания единичной матрицы
def create_identity_matrix_custom(n):

    # Создаёт единичную матрицу размером n x n без использования NumPy.

    identity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1
    return identity_matrix

# 3. Визуализация данных
def visualize_data(X, y):

    # Визуализирует исходные данные: количество автомобилей против прибыли СТО.

    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Зависимость прибыли от количества автомобилей')
    plt.show()

# 4. Версия с использованием NumPy для вычисления функции стоимости и градиентного спуска
def compute_cost_numpy(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)
    return cost

def gradient_descent_numpy(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)
        J_history.append(compute_cost_numpy(X, y, theta))
    return theta, J_history

def predict_numpy(theta, X_new):
    X_new = np.array([1, X_new])
    return X_new.dot(theta)

# 5. Самописные методы для вычисления функции стоимости и градиентного спуска
def compute_cost_custom(X, y, theta):
    m = len(y)
    total_error = 0
    for i in range(m):
        prediction = sum([theta[j] * X[i][j] for j in range(len(theta))])
        total_error += (prediction - y[i]) ** 2
    return (1 / (2 * m)) * total_error

def gradient_descent_custom(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = []
    for _ in range(iterations):
        temp_theta = [0] * len(theta)
        for j in range(len(theta)):
            total_error = 0
            for i in range(m):
                prediction = sum([theta[k] * X[i][k] for k in range(len(theta))])
                total_error += (prediction - y[i]) * X[i][j]
            temp_theta[j] = theta[j] - (alpha / m) * total_error
        theta = temp_theta.copy()
        J_history.append(compute_cost_custom(X, y, theta))
    return theta, J_history

def predict_custom(theta, X_new):
    X_new = [1, X_new]
    prediction = sum([theta[i] * X_new[i] for i in range(len(theta))])
    return prediction

# 6. Запись результатов в файл
def write_to_file(results, filename="output.txt"):
    with open(filename, 'w') as f:
        f.write(results)
