import numpy as np
import matplotlib.pyplot as plt

def create_identity_matrix(n):
    # Способ 1: С использованием стандартной функции
    identity_np = np.eye(n)

    # Способ 2: Без использования стандартной функции
    identity_custom = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    return identity_np, identity_custom

def visualize_data(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Зависимость прибыли СТО от количества автомобилей')
    plt.show()

def compute_cost(X, y, theta):
    m = len(y)  # количество обучающих примеров
    predictions = X.dot(theta)  # предсказания модели
    sq_errors = (predictions - y) ** 2  # квадраты ошибок
    cost = (1 / (2 * m)) * np.sum(sq_errors)
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)  # количество примеров
    J_history = []

    for i in range(iterations):
        predictions = X.dot(theta)  # предсказания
        errors = predictions - y  # ошибки
        theta -= (alpha / m) * X.T.dot(errors)  # обновление theta
        J_history.append(compute_cost(X, y, theta))  # сохраняем историю стоимости

    return theta, J_history

def predict(theta, X_new):
    X_new = np.array([1, X_new])  # добавляем единичный элемент для свободного члена
    return X_new.dot(theta)
