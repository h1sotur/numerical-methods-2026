import numpy as np
import matplotlib.pyplot as plt

#1

def f(x, y):
    return y - x ** 2 + 1


def exact_sol(x):
    return (x + 1) ** 2 - 0.5 * np.exp(x)


# Параметри інтегрування
a = 0.0
b = 2.0
y0 = 0.5
h_fixed = 0.01 
eps = 1e-5

def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


#2

def adams_2nd_order_fixed(a, b, y0, h):
    x_nodes = np.arange(a, b + h / 2, h)
    n_steps = len(x_nodes) - 1
    y_nodes = np.zeros(len(x_nodes))
    y_nodes[0] = y0

    y_nodes[1] = rk4_step(x_nodes[0], y_nodes[0], h)

    y_pred_arr = []
    y_corr_arr = []

    for n in range(1, n_steps):
        x_n = x_nodes[n]
        x_prev = x_nodes[n - 1]
        x_next = x_nodes[n + 1]

        f_n = f(x_n, y_nodes[n])
        f_prev = f(x_prev, y_nodes[n - 1])

        y_pred = y_nodes[n] + (h / 2) * (3 * f_n - f_prev)

        y_corr = y_pred + (5 / 6) * (y_pred - y_nodes[n])
        for _ in range(2):
            y_corr = y_nodes[n] + (h / 2) * (f(x_next, y_corr) + f_n)


        y_final = y_corr - (1 / 6) * (y_corr - y_pred)
        y_nodes[n + 1] = y_final

        y_pred_arr.append(y_pred)
        y_corr_arr.append(y_corr)

    return x_nodes, y_nodes, y_pred_arr, y_corr_arr


#5
def adams_2nd_order_auto(a, b, y0, eps):
    x_points = [a]
    y_points = [y0]
    h_points = []

    h = 0.05

    x_next = a + h
    y_next = rk4_step(a, y0, h)
    x_points.append(x_next)
    y_points.append(y_next)
    h_points.append(h)

    while x_points[-1] < b:
        h_points.append(h)
        x_curr = x_points[-1]
        x_prev = x_points[-2]
        y_curr = y_points[-1]
        y_prev = y_points[-2]

        if x_curr + h > b:
            h = b - x_curr
            if h < 1e-9: break

        f_curr = f(x_curr, y_curr)
        f_prev = f(x_prev, y_prev)

        # Прогноз
        y_pred = y_curr + (h / 2) * (3 * f_curr - f_prev)
        # Корекція
        y_corr = y_pred
        for _ in range(2):
            y_corr = y_curr + (h / 2) * (f(x_curr + h, y_corr) + f_curr)

        error = (1 / 6) * abs(y_corr - y_pred)

        k = 2 ** (2 + 1)
        if error > eps:
            h /= 2
            x_points[-1] = x_points[-2] + h
            y_points[-1] = rk4_step(x_points[-2], y_points[-2], h)
            h_points.pop()
        else:

            y_final = y_corr - (1 / 6) * (y_corr - y_pred)
            x_points.append(x_curr + h)
            y_points.append(y_final)

            if error < eps / k:
                h *= 2

    return np.array(x_points), np.array(y_points), h_points


#6
def runge_kutta_4th_fixed(a, b, y0, h):
    x_nodes = np.arange(a, b + h / 2, h)
    y_nodes = np.zeros(len(x_nodes))
    y_nodes[0] = y0

    for n in range(len(x_nodes) - 1):
        y_nodes[n + 1] = rk4_step(x_nodes[n], y_nodes[n], h)

    return x_nodes, y_nodes


#9
def runge_kutta_4th_auto(a, b, y0, eps):
    x_points = [a]
    y_points = [y0]
    h_points = []

    h = 0.1
    x = a
    y = y0

    while x < b:
        if x + h > b:
            h = b - x


        y_h = rk4_step(x, y, h)


        y_h2_half = rk4_step(x, y, h / 2)
        y_h2 = rk4_step(x + h / 2, y_h2_half, h / 2)


        error = (16 / 15) * abs(y_h - y_h2)

        k = 2 ** (4 + 1)

        if error > eps:
            h /= 2
        else:
            # Крок прийнято
            h_points.append(h)
            x += h
            y = y_h2
            x_points.append(x)
            y_points.append(y)

            if error < eps / k:
                h *= 2

    return np.array(x_points), np.array(y_points), h_points

#1
x_adfixed, y_adfixed, y_pred_ad, y_corr_ad = adams_2nd_order_fixed(a, b, y0, h_fixed)
y_exact_adfixed = exact_sol(x_adfixed)

#3
error_ad_exact = y_adfixed - y_exact_adfixed

#4

error_ad_estimated = (1 / 6) * (np.array(y_corr_ad) - np.array(y_pred_ad))[1:]

#2
x_adauto, y_adauto, h_adauto = adams_2nd_order_auto(a, b, y0, eps)

#3
x_rkfixed, y_rkfixed = runge_kutta_4th_fixed(a, b, y0, h_fixed)
y_exact_rkfixed = exact_sol(x_rkfixed)

#7
error_rk_exact = y_rkfixed - y_exact_rkfixed

#8

_, y_rk_half = runge_kutta_4th_fixed(a, b, y0, h_fixed / 2)
error_rk_runge = (16 / 15) * (y_rk_half[::2] - y_rkfixed)

# 4
x_rkauto, y_rkauto, h_rkauto = runge_kutta_4th_auto(a, b, y0, eps)

plt.figure(figsize=(14, 10))

#3 4
plt.subplot(2, 2, 1)
plt.plot(x_adfixed, error_ad_exact, 'b-', label='Точна похибка (Завдання 3)')
x_ad_estimated_plots = x_adfixed[-len(error_ad_estimated):]
plt.plot(x_ad_estimated_plots, error_ad_estimated, 'r--', label='Оцінка (y_corr - y_pred) (Завдання 4)')
plt.title('Похибки методу Адамса (Фіксований крок)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend()

# 5
plt.subplot(2, 2, 2)
# ДИНАМІЧНЕ ВИРІВНЮВАННЯ
x_ad_auto_plots = x_adauto[:len(h_adauto)]
plt.step(x_ad_auto_plots, h_adauto, 'g-', where='post', label='Величина кроку h(x)')
plt.title('Залежність кроку h(x) від x для Адамса ')
plt.xlabel('x')
plt.ylabel('h')
plt.grid(True)
plt.legend()

#7 8
plt.subplot(2, 2, 3)
plt.plot(x_rkfixed, error_rk_exact, 'b-', label='Точна похибка ')
plt.plot(x_rkfixed, error_rk_runge, 'r--', label='Оцінка за Рунге ')
plt.title('Похибки Рунге-Кутта 4-го порядку (Фіксований крок)')
plt.xlabel('x')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend()

#9
plt.subplot(2, 2, 4)
# ДИНАМІЧНЕ ВИРІВНЮВАННЯ
x_rk_auto_plots = x_rkauto[:len(h_rkauto)]
plt.step(x_rk_auto_plots, h_rkauto, 'm-', where='post', label='Величина кроку h(x)')
plt.title('Залежність кроку h(x) від x для РК4 ')
plt.xlabel('x')
plt.ylabel('h')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Метод Адамса (фіксований крок): Макс. точна похибка = {np.max(np.abs(error_ad_exact)):.2e}")
print(f"Метод Рунге-Кутта (фіксований крок): Макс. точна похибка = {np.max(np.abs(error_rk_exact)):.2e}")
print(f"Автоматичний крок Адамса: Кількість точок інтегрування = {len(x_adauto)}")
print(f"Автоматичний крок РК4: Кількість точок інтегрування = {len(x_rkauto)}")
print("Всі графіки побудовано успішно. Лабораторна робота виконана.")