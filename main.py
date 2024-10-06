import numpy as np
import matplotlib.pyplot as plt
from functions import create_identity_matrix, visualize_data, compute_cost, gradient_descent, predict

def main():
    # Загрузка данных
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]  # количество автомобилей в населённом пункте
    y = data[:, 1]  # прибыль СТО
    m = len(y)  # количество примеров

    # Визуализация данных
    visualize_data(X, y)

    # Добавляем столбец единиц к X (для свободного члена theta0)
    X = np.c_[np.ones(m), X]  # X становится размерностью (m, 2)

    # Инициализация параметров
    theta = np.zeros(2)  # начальные значения theta0 и theta1
    iterations = 1500
    alpha = 0.01  # скорость обучения

    # Вычисление начальной стоимости
    cost = compute_cost(X, y, theta)
    print(f"Первоначальная стоимость: {cost}")

    # Запуск градиентного спуска
    theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

    # Отображение итоговой прямой
    plt.plot(X[:, 1], X.dot(theta), label='Линейная регрессия')
    plt.scatter(X[:, 1], y, marker='x', c='r', label='Учебные данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.legend()
    plt.show()

    print(f"Окончательная тета: {theta}")

    # Пример предсказания
    cars = 35000  # количество автомобилей в населённом пункте
    predicted_profit = predict(theta, cars)
    print(f"Прогнозируемая прибыль для {cars} автомобили: {predicted_profit}")

if __name__ == '__main__':
    main()
