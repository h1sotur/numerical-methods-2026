import numpy as np


#1
def generate_data(n=100, x_val=2.5):
    A = np.random.uniform(1, 100, (n, n))
    x_true = np.full(n, x_val)
    # Обчислюємо b_i = sum(a_ij * x_j) [cite: 41]
    B = A @ x_true

    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_B.txt', B)
    print(f"Матрицю A та вектор B збережено (n={n}).")


#2
def load_data():
    A = np.loadtxt('matrix_A.txt')
    B = np.loadtxt('vector_B.txt')
    return A, B


def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)

    for k in range(n):

        for i in range(k, n):
            L[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(k))

        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(k))) / L[k, k]

    return L, U


def solve_lu(L, U, B):
    n = len(L)
    z = np.zeros(n)
    for k in range(n):
        z[k] = (B[k] - sum(L[k, j] * z[j] for j in range(k))) / L[k, k]

    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = z[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))
    return x


def vector_norm(v):
    return np.max(np.abs(v))


if __name__ == "__main__":

    n_dim = 100
    generate_data(n_dim)

    A, B = load_data()
    L, U = lu_decomposition(A)
    np.savetxt('matrix_LU.txt', np.add(L, U) - np.eye(n_dim))

    x_0 = solve_lu(L, U, B)

    initial_error = vector_norm(A @ x_0 - B)
    print(f"Початкова точність (eps): {initial_error:.2e}")

    eps_target = 1e-14
    x_current = x_0.copy()
    iteration = 0

    while True:
        R = B - (A @ x_current)

        if vector_norm(R) <= eps_target:
            break

        delta_x = solve_lu(L, U, R)
        x_current = x_current + delta_x
        iteration += 1

        if iteration > 20:  # Запобіжник
            break

    print(f"Уточнений розв'язок знайдено за {iteration} ітерацій.")
    print(f"Кінцева нев'язка: {vector_norm(A @ x_current - B):.2e}")