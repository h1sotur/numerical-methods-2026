import numpy as np
import matplotlib.pyplot as plt
import csv
import math

#2
def read_data(filename):
    n_val, t_val = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row['n'] or not row['t']: continue
            n_val.append(float(row['n']))
            t_val.append(float(row['t']))
    return np.array(n_val), np.array(t_val)

def get_divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]

def newton_calc(coef, x_data, x):
    n = len(x_data) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p
#3
def factorial_calc(y, x, target_x):
    h = x[1] - x[0]
    t = (target_x - x[0]) / h
    n = len(y)
    diffs = np.zeros([n, n])
    diffs[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diffs[i][j] = diffs[i + 1][j - 1] - diffs[i][j - 1]

    res = diffs[0, 0]
    term = 1
    for k in range(1, n):
        term *= (t - (k - 1))
        # ВИПРАВЛЕНО: використовуємо math.factorial замість np.math.factorial
        res += (term * diffs[0, k]) / math.factorial(k)
    return res

x_nodes, y_nodes = read_data("data.csv")
c = get_divided_diff(x_nodes, y_nodes)

target = 6000
res_newton = newton_calc(c, x_nodes, target)
res_fact = factorial_calc(y_nodes, x_nodes, target)

print(f"Прогноз для n={target}:")
print(f"Ньютон: {res_newton:.2f} мс")
print(f"Факторіальний: {res_fact:.2f} мс")
#3 4
x_fine = np.linspace(min(x_nodes), max(x_nodes), 100)
y_fine = [newton_calc(c, x_nodes, val) for val in x_fine]

x_tab = np.linspace(min(x_nodes), max(x_nodes), 20)

with open("tabulation_results.txt", "w", encoding="utf-8") as f_out:
    # Заголовок файлу згідно з вимогами [cite: 532, 535]
    f_out.write("№  | x (Вузол)   | Nn(x) (Прогноз) | Похибка eps(x)\n")
    f_out.write("-" * 50 + "\n")

    for i, val in enumerate(x_tab):
        y_interp = newton_calc(c, x_nodes, val)  # Значення многочлена Ньютона [cite: 535]
        # Похибка як різниця між методами або відхилення [cite: 535]
        err = abs(y_interp - factorial_calc(y_nodes, x_nodes, val))

        # Записуємо рядок даних у файл
        line = f"{i:<2} | {val:<11.2f} | {y_interp:<15.2f} | {err:<14.2e}\n"
        f_out.write(line)

print("Файл 'tabulation_results.txt' успішно створено.")

plt.figure(figsize=(10, 5))
plt.plot(x_nodes, y_nodes, 'ro', label='Точки')
plt.plot(x_fine, y_fine, 'b-', label='Ньютон/Факторіал')
plt.scatter([target], [res_newton], color='green', label=f'Прогноз {target}')
plt.title("Інтерполяція (Варіант 1)")
plt.legend()
plt.grid(True)
plt.show()