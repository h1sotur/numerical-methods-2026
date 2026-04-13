import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


#1
def f(x):
    # f(x) = 50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x-12)^2)
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24
target_eps = 1e-12

#2
I0, _ = quad(f, a, b)
print(f"Точне значення I0: {I0:.15f}")


#3
def simpson(f, a, b, N):
    if N % 2 != 0: N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    res = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    return res


#4
N_range = np.arange(10, 1001, 10)
errors_N = [abs(simpson(f, a, b, n) - I0) for n in N_range]

N_opt = 0
for n in range(10, 5000, 2):
    if abs(simpson(f, a, b, n) - I0) <= target_eps:
        N_opt = n
        break
print(f"N_opt: {N_opt}, Точність: {abs(simpson(f, a, b, N_opt) - I0):.2e}")

#5-7
N0 = (N_opt // 10)
if N0 < 8: N0 = 8
N0 = N0 + (8 - N0 % 8) if N0 % 8 != 0 else N0

I_h = simpson(f, a, b, N0)
I_2h = simpson(f, a, b, N0 // 2)
I_4h = simpson(f, a, b, N0 // 4)

I_R = I_h + (I_h - I_2h) / 15
I_E = (I_2h ** 2 - I_h * I_4h) / (2 * I_2h - (I_h + I_4h))
p_val = (1 / np.log(2)) * np.log(abs((I_4h - I_2h) / (I_2h - I_h)))

print(f"\nРезультати для N0={N0}:")
print(f"Похибка Сімпсона: {abs(I_h - I0):.2e}")
print(f"Похибка Рунге-Ромберга: {abs(I_R - I0):.2e}")
print(f"Похибка Ейткена: {abs(I_E - I0):.2e}")
print(f"Порядок точності p: {p_val:.2f}")


#8-9
def adaptive_simpson(f, a, b, delta, storage):
    storage['calls'] += 5
    c = (a + b) / 2
    h = b - a
    fa, fb, fc = f(a), f(b), f(c)
    fl, fr = f((a + c) / 2), f((c + b) / 2)

    for p in [a, (a + c) / 2, c, (c + b) / 2, b]:
        storage['points'].add(p)

    I1 = (h / 6) * (fa + 4 * fc + fb)
    I2 = (h / 12) * (fa + 4 * fl + 2 * fc + 4 * fr + fb)

    if abs(I1 - I2) <= 15 * delta:
        return I2 + (I2 - I1) / 15
    else:
        return (adaptive_simpson(f, a, c, delta / 2, storage) +
                adaptive_simpson(f, c, b, delta / 2, storage))


deltas = np.logspace(-1, -7, 7)
eps_list, calls_list = [], []

for d in deltas:
    st = {'calls': 0, 'points': set()}
    val = adaptive_simpson(f, a, b, d, st)
    eps_list.append(abs(val - I0))
    calls_list.append(st['calls'])

plt.figure(figsize=(10, 4))
plt.semilogy(N_range, errors_N, label='Simpson Error')
plt.axhline(y=target_eps, color='r', ls='--', label='1e-12')
plt.title('Залежність похибки від N')
plt.grid(True, which='both')
plt.legend()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.set_xlabel('Параметр дельта (δ)')
ax1.set_ylabel('Абсолютна похибка ε', color='purple')
ax1.loglog(deltas, eps_list, 'o-', color='purple')
ax1.invert_xaxis()

ax2 = ax1.twinx()
ax2.set_ylabel('Виклики функції f(x)', color='orange')
ax2.loglog(deltas, calls_list, 's--', color='orange')

plt.title('Аналіз адаптивного алгоритму (ε та виклики від δ)')
ax1.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()