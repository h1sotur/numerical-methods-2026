import numpy as np
import matplotlib.pyplot as plt


#1

def f1(x):
    return x[0] ** 2 + x[1] ** 2 - 4


def f2(x):
    return x[0] - x[1] ** 2 + 1


def plot_system():
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)

    Z1 = X ** 2 + Y ** 2 - 4
    Z2 = X - Y ** 2 + 1

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z1, levels=[0], colors='r')
    plt.contour(X, Y, Z2, levels=[0], colors='b')
    plt.grid(True)
    plt.title("Графіки системи нелінійних рівнянь")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["f1(x)=0", "f2(x)=0"])
    plt.show()


#4
def target_function(x):
    return f1(x) ** 2 + f2(x) ** 2


#3
def rosenbrock(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


#2
def hooke_jeeves(func, start_point, step_size, eps1, eps2, q=2, p=2, max_iter=1000):
    x0 = np.array(start_point, dtype=float)
    delta = np.array(step_size, dtype=float)
    trajectory = [x0.copy()]

    n = len(x0)
    k = 0

    while k < max_iter:
        x1 = x0.copy()

        for i in range(n):
            x_trial = x1.copy()
            x_trial[i] += delta[i]
            if func(x_trial) < func(x1):
                x1[i] = x_trial[i]
            else:
                x_trial[i] -= 2 * delta[i]
                if func(x_trial) < func(x1):
                    x1[i] = x_trial[i]
                else:
                    delta[i] /= q

        if np.all(delta < eps1) or abs(func(x1) - func(x0)) < eps2:
            return x1, trajectory

        if np.array_equal(x1, x0):
            if np.all(delta < eps1):
                return x0, trajectory
            continue

        while True:
            xp = x1 + p * (x1 - x0)

            x2 = xp.copy()
            for i in range(n):
                x_trial = x2.copy()
                x_trial[i] += delta[i]
                if func(x_trial) < func(x2):
                    x2[i] = x_trial[i]
                else:
                    x_trial[i] -= 2 * delta[i]
                    if func(x_trial) < func(x2):
                        x2[i] = x_trial[i]

            if func(x2) < func(x1):
                x0 = x1.copy()
                x1 = x2.copy()
                trajectory.append(x1.copy())
            else:
                x0 = x1.copy()  
                break

        k += 1

    return x1, trajectory

plot_system()

start_sys = [1.0, 1.0]      # Початкове наближення X(0)
delta_init = [0.1, 0.1]    # Початкова величина кроку
eps1 = 1e-6                 # Критерій закінчення 1
eps2 = 1e-6                 # Критерій закінчення 2
q_val = 2                   # Коефіцієнт зміни кроку
p_val = 2                   # Коефіцієнт пошуку по зразку

print(f"Параметри пошуку")
print(f"Початкова точка X(0): {start_sys}")
print(f"Початковий крок delta: {delta_init}")
print(f"Коефіцієнти: q = {q_val}, p = {p_val}")
print(f"Точність: eps1 = {eps1}, eps2 = {eps2}\n")

result, trajectory = hooke_jeeves(target_function, start_sys, delta_init, eps1, eps2, q=q_val, p=p_val)

print("Результати розв'язку")
print(f"Розв'язок системи: x1 = {result[0]:.6f}, x2 = {result[1]:.6f}")
print(f"Значення цільової функції Ф(X): {target_function(result):.10f}")
print(f"Число кроків на траєкторії спуску: {len(trajectory)}") #

with open("trajectory.txt", "w", encoding="utf-8") as f:
    f.write("Координати точок траєкторії спуску:\n") #
    for i, point in enumerate(trajectory):
        f.write(f"Крок {i}: x1 = {point[0]:.6f}, x2 = {point[1]:.6f}\n")

print("\nТраєкторію спуску збережено у файл 'trajectory.txt'.")