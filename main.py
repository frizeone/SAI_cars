import numpy as np
import matplotlib.pyplot as plt
from functions import (
    create_identity_matrix_numpy, create_identity_matrix_custom, visualize_data,
    compute_cost_numpy, gradient_descent_numpy, predict_numpy,
    compute_cost_custom, gradient_descent_custom, predict_custom, write_to_file
)

def main():
    # Спрашиваем у пользователя размерность матрицы
    n = int(input("Введите размерность единичной матрицы: "))

    # Спрашиваем у пользователя, какую версию для создания единичной матрицы использовать
    version_matrix = input("Выберите версию для создания матрицы (1 - с использованием NumPy, 2 - самописная): ")

    if version_matrix == '1':
        identity_matrix = create_identity_matrix_numpy(n)
    elif version_matrix == '2':
        identity_matrix = create_identity_matrix_custom(n)
    else:
        print("Некорректный выбор.")
        return

    print(f"Созданная единичная матрица размером {n}x{n}:")
    for row in identity_matrix:
        print(row)

    # Загрузка данных
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]  # количество автомобилей в населённом пункте
    y = data[:, 1]  # прибыль СТО
    m = len(y)

    # Визуализация данных
    visualize_data(X, y)

    # Добавляем столбец единиц к X (для свободного члена theta0)
    X_with_ones = np.c_[np.ones(m), X]

    # Инициализация параметров
    theta = np.zeros(2)
    iterations = 1500
    alpha = 0.01

    # Запрашиваем у пользователя, какую версию использовать для вычислений
    version = input("Выберите версию (1 - с использованием NumPy, 2 - самописные методы): ")

    if version == '1':
        # Использование NumPy для расчётов
        cost = compute_cost_numpy(X_with_ones, y, theta)
        theta, J_history = gradient_descent_numpy(X_with_ones, y, theta, alpha, iterations)
        predicted_profit = predict_numpy(theta, 35000)

        results = f"Первоначальные затраты (NumPy): {cost}\nКонечная тета (NumPy): {theta}\n"
        results += f"Прогнозируемая прибыль для 35 000 автомобилей (NumPy): {predicted_profit}\n"

    elif version == '2':
        # Использование самописных методов
        X_custom = X_with_ones.tolist()
        y_custom = y.tolist()
        theta_custom = [0, 0]

        cost = compute_cost_custom(X_custom, y_custom, theta_custom)
        theta, J_history = gradient_descent_custom(X_custom, y_custom, theta_custom, alpha, iterations)
        predicted_profit = predict_custom(theta, 35000)

        results = f"Первоначальная стоимость (пользовательская): {cost}\nОкончательная тета (пользовательская): {theta}\n"
        results += f"Прогнозируемая прибыль для 35 000 автомобилей (пользовательская): {predicted_profit}\n"

    else:
        print("Некорректный выбор версии.")
        return

    # Печать результатов и запись в файл
    print(results)
    write_to_file(results)

    # Построение графика изменения функции стоимости
    plt.plot(range(iterations), J_history, label='Функция затрат')
    plt.xlabel('Итерации')
    plt.ylabel('Стоимость')
    plt.title('Функция стоимости за итерации')
    plt.legend()
    plt.show()

    # Построение графика итоговой прямой
    plt.plot(X_with_ones[:, 1], X_with_ones.dot(theta), label='Линейная регрессия')
    plt.scatter(X_with_ones[:, 1], y, marker='x', c='r', label='Учебные данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.legend()
    plt.show()

    # Пример предсказания
    cars = 35000
    print(f"Прогнозируемая прибыль для {cars} автомобили: {predicted_profit}")

if __name__ == '__main__':
    main()
