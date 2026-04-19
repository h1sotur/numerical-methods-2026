import numpy as np


def run_laboratory_work():
    #1
    n = 100
    eps_target = 1e-14

    A = np.random.uniform(0, 10, (n, n))
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 10)

    x_exact = np.full(n, 2.5)
    b = A @ x_exact

    np.savetxt("matrix_A.txt", A)
    np.savetxt("vector_B.txt", b)
    def load_data():
        return np.loadtxt("matrix_A.txt"), np.loadtxt("vector_B.txt")

    def vector_norm(v):
        return np.max(np.abs(v))

    def matrix_norm(M):
        return np.max(np.sum(np.abs(M), axis=1))

    def simple_iteration(A, b, eps):
        n = len(b)
        tau = 1.0 / matrix_norm(A)
        x = np.ones(n)
        for k in range(10000):
            x_new = x - tau * (A @ x - b)
            if vector_norm(x_new - x) < eps:
                return x_new, k + 1
            x = x_new
        return x, 10000

    def jacobi_method(A, b, eps):
        n = len(b)
        x = np.ones(n)
        D = np.diag(A)
        R = A - np.diag(D)
        for k in range(10000):
            x_new = (b - R @ x) / D
            if vector_norm(x_new - x) < eps:
                return x_new, k + 1
            x = x_new
        return x, 10000

    def seidel_method(A, b, eps):
        n = len(b)
        x = np.ones(n)
        for k in range(10000):
            x_old = x.copy()
            for i in range(n):
                sum1 = np.dot(A[i, :i], x[:i])
                sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
                x[i] = (b[i] - sum1 - sum2) / A[i, i]
            if vector_norm(x - x_old) < eps:
                return x, k + 1
        return x, 10000

    #4
    A_loaded, b_loaded = load_data()

    methods = [
        ("Проста ітерація", simple_iteration),
        ("Якобі", jacobi_method),
        ("Зейдель", seidel_method)
    ]

    print(f"\n4: результати розв'язку (eps = {eps_target})")
    print("-" * 60)
    for name, method_func in methods:
        solution, iters = method_func(A_loaded, b_loaded, eps_target)
        error = vector_norm(solution - 2.5)
        print(f"{name:16} | Ітерацій: {iters:4} | Похибка: {error:.2e}")

if __name__ == "__main__":
    run_laboratory_work()