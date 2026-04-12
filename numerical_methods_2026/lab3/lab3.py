import numpy as np
import matplotlib.pyplot as plt
import csv
#2

def form_matrix(x, m):
    n_size = m + 1
    matrix = np.zeros((n_size, n_size))
    for k in range(n_size):
        for l in range(n_size):
            matrix[k, l] = np.sum(x ** (k + l))
    return matrix


def form_vector(x, y, m):
    n_size = m + 1
    vector = np.zeros(n_size)
    for k in range(n_size):
        vector[k] = np.sum(y * (x ** k))
    return vector


def gauss_solve(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    #прямий
    for k in range(n):
        max_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    #зворотній
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]
    return x_sol


def polynomial(x, coef):
    y_approx = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_approx += c * (x ** i)
    return y_approx


def calculate_variance(y_true, y_approx):
    return np.sqrt(np.mean((y_true - y_approx) ** 2))


# 1
months = np.arange(1, 25)
temps = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
                  -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3])

#3
variances = []
degrees = range(1, 11)

for m in degrees:
    A_mat = form_matrix(months, m)
    B_vec = form_vector(months, temps, m)
    coefficients = gauss_solve(A_mat, B_vec)
    y_pred = polynomial(months, coefficients)
    var = calculate_variance(temps, y_pred)
    variances.append(var)
    print(f"Ступінь m={m}: Дисперсія = {var:.4f}")

optimal_m = degrees[np.argmin(variances)]
print(f"\nОптимальний ступінь за мінімумом дисперсії: {optimal_m}")


final_A = form_matrix(months, optimal_m)
final_B = form_vector(months, temps, optimal_m)
final_coef = gauss_solve(final_A, final_B)

#5  6
x_future = np.array([25, 26, 27])
y_future = polynomial(x_future, final_coef)
print(f"Прогноз на місяці 25, 26, 27: {y_future}")

#4  7
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.scatter(months, temps, color='red', label='Фактичні дані')
x_fine = np.linspace(1, 27, 200)  # Для гладкості графіка
plt.plot(x_fine, polynomial(x_fine, final_coef), label=f'Поліном (m={optimal_m})')
plt.scatter(x_future, y_future, color='green', marker='x', label='Прогноз')
plt.title('Апроксимація температури методом найменших квадратів')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
y_approx_all = polynomial(months, final_coef)
error = np.abs(temps - y_approx_all)  # [cite: 81]
plt.bar(months, error, color='gray', alpha=0.7, label='Абсолютна похибка')
plt.title('Графік похибки апроксимації за місяцями')
plt.xlabel('Місяць')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()