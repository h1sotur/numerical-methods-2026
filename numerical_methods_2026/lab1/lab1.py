import requests
import numpy as np
import matplotlib.pyplot as plt



def get_elevation_data():
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    response = requests.get(url)
    return response.json()["results"]



def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  # [cite: 129]



def solve_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    alfa = np.zeros(n + 1)
    beta = np.ones(n + 1)
    hamma = np.zeros(n + 1)
    delta = np.zeros(n + 1)

    for i in range(1, n):
        alfa[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        hamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])


    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    for i in range(1, n):
        m = alfa[i] * A[i - 1] + beta[i]
        A[i] = -hamma[i] / m
        B[i] = (delta[i] - alfa[i] * B[i - 1]) / m


    c = np.zeros(n + 1)
    for i in range(n - 1, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]


    a = y[:-1]
    d = np.diff(c) / (3 * h)
    b = (np.diff(y) / h) - (h / 3) * (c[1:] + 2 * c[:-1])

    return a, b, c[:-1], d, h



results = get_elevation_data()
coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]


distances = [0]
for i in range(1, len(coords)):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)


plt.figure(figsize=(12, 7))
plt.scatter(distances, elevations, color='black', label='Реальні дані GPS')

for n_nodes in [10, 15, 20]:
    idx = np.linspace(0, len(distances) - 1, n_nodes, dtype=int)
    x_nodes = np.array(distances)[idx]
    y_nodes = np.array(elevations)[idx]

    a, b, c, d, h = solve_spline(x_nodes, y_nodes)

    x_fine = np.linspace(distances[0], distances[-1], 200)
    y_spline = []
    for val in x_fine:
        i = np.searchsorted(x_nodes, val) - 1
        i = max(0, min(i, len(a) - 1))
        dx = val - x_nodes[i]
        y_spline.append(a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3)

    plt.plot(x_fine, y_spline, label=f'Сплайн ({n_nodes} вузлів)')

#додатково
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
energy = 80 * 9.81 * total_ascent

print(f"Загальна довжина: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")
print(f"Механічна робота: {energy / 1000:.2f} кДж")

plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль маршруту на Говерлу (Кубічний сплайн)")
plt.legend()
plt.grid(True)
plt.show()


with open("input_data.txt", "w", encoding="utf-8") as f_in:
    f_in.write("№  | Latitude  | Longitude | Elevation (m) | Distance (m)\n")
    f_in.write("-" * 65 + "\n")
    for i in range(len(results)):
        line = (f"{i:<2} | {results[i]['latitude']:.6f} | "
                f"{results[i]['longitude']:.6f} | "
                f"{elevations[i]:.2f} | "
                f"{distances[i]:.2f}\n")
        f_in.write(line)

print("Табуляцію записано у файл input_data.txt")