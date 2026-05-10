import math
import numpy as np


def F(x):
    return math.sin(x) - 0.5 * x


def dF(x):
    return math.cos(x) - 0.5


def d2F(x):
    return -math.sin(x)


def check_stop(x_new, x_old, eps):
    return abs(F(x_new)) < eps and abs(x_new - x_old) < eps


#1
def tabulate_and_find_roots(a, b, h=0.1):
    x_vals = np.arange(a, b + h, h)
    roots_approx = []

    with open("tabulation.txt", "w") as f:
        for i in range(len(x_vals)):
            y = F(x_vals[i])
            f.write(f"{x_vals[i]:.2f}\t{y:.4f}\n")

            if i > 0 and F(x_vals[i - 1]) * y < 0:
                roots_approx.append((x_vals[i - 1] + x_vals[i]) / 2)
    return roots_approx


#2

def simple_iteration(x0, eps, tau=-1.0):
    x_prev = x0
    for i in range(1, 101):
        x_next = x_prev + tau * F(x_prev)
        if check_stop(x_next, x_prev, eps): return x_next, i
        x_prev = x_next
    return x_prev, 100


def newton_method(x0, eps):
    x_prev = x0
    for i in range(1, 101):
        x_next = x_prev - F(x_prev) / dF(x_prev)
        if check_stop(x_next, x_prev, eps): return x_next, i
        x_prev = x_next
    return x_prev, 100


def chebyshev_method(x0, eps):
    x_prev = x0
    for i in range(1, 101):
        f, df, d2f = F(x_prev), dF(x_prev), d2F(x_prev)
        x_next = x_prev - f / df - 0.5 * (f ** 2 * d2f) / (df ** 3)
        if check_stop(x_next, x_prev, eps): return x_next, i
        x_prev = x_next
    return x_prev, 100


def secant_method(x0, x1, eps):
    x_prev2, x_prev1 = x0, x1
    for i in range(1, 101):
        f1, f2 = F(x_prev1), F(x_prev2)
        x_next = x_prev1 - f1 * (x_prev1 - x_prev2) / (f1 - f2)
        if check_stop(x_next, x_prev1, eps): return x_next, i
        x_prev2, x_prev1 = x_prev1, x_next
    return x_prev1, 100


def parabola_method(x0, x1, x2, eps):
    def div_diff(xa, xb):
        return (F(xa) - F(xb)) / (xa - xb)

    xn, xn1, xn2 = x2, x1, x0
    for i in range(1, 101):
        f12 = div_diff(xn, xn1)
        f123 = (f12 - div_diff(xn1, xn2)) / (xn - xn2)
        w = f12 + (xn - xn1) * f123
        det = math.sqrt(max(0, w ** 2 - 4 * F(xn) * f123))
        delta = -2 * F(xn) / (w + det if w > 0 else w - det)
        x_next = xn + delta
        if check_stop(x_next, xn, eps): return x_next, i
        xn2, xn1, xn = xn1, xn, x_next
    return xn, 100


def reverse_interpolation(x0, x1, x2, eps):
    xn, xn1, xn2 = x2, x1, x0
    for i in range(1, 101):
        y, y1, y2 = F(xn), F(xn1), F(xn2)
        x_next = (y1 * y / ((y2 - y1) * (y2 - y))) * xn2 + \
                 (y2 * y / ((y1 - y2) * (y1 - y))) * xn1 + \
                 (y2 * y1 / ((y - y2) * (y - y1))) * xn
        if check_stop(x_next, xn, eps): return x_next, i
        xn2, xn1, xn = xn1, xn, x_next
    return xn, 100


#6-7
def save_coeffs(coeffs):
    with open("coeffs.txt", "w") as f:
        f.write(" ".join(map(str, coeffs)))


def read_coeffs():
    with open("coeffs.txt", "r") as f:
        return list(map(float, f.read().split()))


#8
def horner_eval(a, x):
    m = len(a) - 1
    b = [0.0] * (m + 1)
    b[m] = a[m]
    for i in range(m - 1, -1, -1):
        b[i] = a[i] + x * b[i + 1]

    c = [0.0] * m
    c[m - 1] = b[m]
    for i in range(m - 2, 0, -1):
        c[i] = b[i + 1] + x * c[i + 1]
    return b[0], c[1]


def algebraic_newton(a, x0, eps):
    x_prev = x0
    for i in range(1, 101):
        f_val, df_val = horner_eval(a, x_prev)
        x_next = x_prev - f_val / df_val
        if abs(x_next - x_prev) < eps: return x_next, i
        x_prev = x_next
    return x_prev, 100


#9
def lin_method(a, alpha0, beta0, eps):
    m = len(a) - 1
    alpha, beta = alpha0, beta0
    for i in range(1, 201):
        p, q = -2 * alpha, alpha ** 2 + beta ** 2
        b = [0.0] * (m + 1)
        b[m] = a[m]
        b[m - 1] = a[m - 1] + p * b[m]
        for j in range(m - 2, 1, -1):
            b[j] = a[j] + p * b[j + 1] + q * b[j + 2]

        q_new = a[0] / b[2]
        p_new = (a[1] * b[2] - a[0] * b[3]) / (b[2] ** 2)
        alpha_new = -p_new / 2
        beta_new = math.sqrt(abs(q_new - alpha_new ** 2))

        if abs(alpha_new - alpha) < eps and abs(beta_new - beta) < eps:
            return alpha_new, beta_new, i
        alpha, beta = alpha_new, beta_new
    return alpha, beta, 200


if __name__ == "__main__":
    #4
    eps = 1e-10

    roots = tabulate_and_find_roots(-5, 5)
    print(f"Знайдено наближені корені: {roots}")

    if roots:
        for i, r in enumerate(roots[:2]):
            print(f"\n--- Уточнення кореня №{i + 1} (x ≈ {r:.4f}) ---")
            print(f"1. Проста ітерація:  {simple_iteration(r, eps)}")
            print(f"2. Метод Ньютона:    {newton_method(r, eps)}")
            print(f"3. Метод Чебишева:  {chebyshev_method(r, eps)}")
            print(f"4. Метод хорд:      {secant_method(r - 0.1, r + 0.1, eps)}")
            print(f"5. Метод парабол:   {parabola_method(r - 0.1, r, r + 0.1, eps)}")
            print(f"6. Зворотна інтерп.: {reverse_interpolation(r - 0.1, r, r + 0.1, eps)}")

    alg_coeffs = [-1.0, 1.0, -1.0, 1.0]  # a0, a1, a2, a3
    save_coeffs(alg_coeffs)

    c = read_coeffs()
    real_root, n_iters = algebraic_newton(c, 1.5, eps)
    print(f"\nДійсний корінь (Горнер): {real_root:.6f}, ітерацій: {n_iters}")

    re, im, l_iters = lin_method(c, 0.1, 0.9, eps)
    print(f"Комплексні корені (Лін): {re:.6f} ± {im:.6f}i, ітерацій: {l_iters}")