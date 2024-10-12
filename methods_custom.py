import matplotlib.pyplot as plt

# Самописная функция для создания единичной матрицы
def create_identity_matrix_custom(n):
    identity_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        identity_matrix[i][i] = 1
    return identity_matrix

# Визуализация данных
def visualize_data(X, y):
    plt.scatter(X, y, marker='x', c='r')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль СТО')
    plt.title('Зависимость прибыли от количества автомобилей')
    plt.show()

# Самописная функция для вычисления стоимости
def compute_cost_custom(X, y, theta):
    m = len(y)
    total_error = 0
    for i in range(m):
        prediction = sum([theta[j] * X[i][j] for j in range(len(theta))])
        total_error += (prediction - y[i]) ** 2
    return (1 / (2 * m)) * total_error

# Самописная функция для градиентного спуска
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
        theta = [temp_theta[i] for i in range(len(temp_theta))]  # Ручное копирование
        J_history.append(compute_cost_custom(X, y, theta))
    return theta, J_history

# Самописная функция для предсказания
def predict_custom(theta, X_new):
    X_new = [1, X_new]  # Добавляем 1 для theta_0
    prediction = sum([theta[i] * X_new[i] for i in range(len(theta))])
    return prediction

# Функция для записи результатов в файл
def write_to_file(results, filename="output.txt"):
    with open(filename, 'w') as f:
        f.write(results)
