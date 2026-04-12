import numpy as np
import matplotlib.pyplot as plt

# 1.
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_dt_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0
exact_val = dM_dt_exact(t0)

# 2.
h_range = np.logspace(-20, 3, 500)
errors_h = []
for h in h_range:
    y_approx = (M(t0 + h) - M(t0 - h)) / (2 * h)
    errors_h.append(abs(y_approx - exact_val))

h0 = h_range[np.argmin(errors_h)] # Оптимальне h0

# 3-5.
h_fix = 1e-3
yh = (M(t0 + h_fix) - M(t0 - h_fix)) / (2 * h_fix)
y2h = (M(t0 + 2 * h_fix) - M(t0 - 2 * h_fix)) / (4 * h_fix)
R1 = abs(yh - exact_val)

# 6.
y_RR = yh + (yh - y2h) / 3
R2 = abs(y_RR - exact_val)

# 7.
y4h = (M(t0 + 4 * h_fix) - M(t0 - 4 * h_fix)) / (8 * h_fix)
y_E = (y2h**2 - y4h * yh) / (2 * y2h - (y4h + yh))
p_aitken = (1 / np.log(2)) * np.log(abs((y4h - y2h) / (y2h - yh)))
R3 = abs(y_E - exact_val)


fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3)

t_plot = np.linspace(0, 20, 500)
axs[0, 0].plot(t_plot, M(t_plot), color='royalblue', label='M(t)')
axs[0, 0].set_title("Функція M(t)")
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(t_plot, dM_dt_exact(t_plot), color='seagreen', label="M'(t) аналітична")
axs[0, 1].axvline(t0, color='red', linestyle='--', label=f't0={t0}')
axs[0, 1].set_title("Похідна M'(t)")
axs[0, 1].grid(True, alpha=0.3)

#2
axs[1, 0].loglog(h_range, errors_h, 'r.-', markersize=3)
axs[1, 0].set_title("Похибка від кроку h")
axs[1, 0].set_xlabel("h (зменшення)")
axs[1, 0].set_ylabel("|E(h)|")
axs[1, 0].grid(True, which="both", alpha=0.2)

#6-7
labels = ['Центр. різниця', 'Рунге-Ромберг', 'Ейткен']
errs = [R1, R2, R3]
axs[1, 1].bar(labels, errs, color=['orange', 'skyblue', 'lightgreen'])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("Порівняння похибок методів")
for i, v in enumerate(errs):
    axs[1, 1].text(i, v, f'{v:.1e}', ha='center', va='bottom')

plt.suptitle("Чисельне диференціювання M(t)", fontsize=16)
plt.show()

# Друк в консоль для звіту
print(f"Точне значення: {exact_val}")
print(f"Порядок точності p (Ейткен): {p_aitken:.2f}")