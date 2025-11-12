# Adaptive Integration

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(x)

def calculate_I_coarse(func, a, b):
    c = (a + b) / 2
    dx = b - a
    return (dx / 6) * (func(a) + 4 * func(c) + func(b))

errors = []
interval_sizes = []
intervals_final = []
function_evaluations = 0


def adaptive_integration(func, a, b, tol, I_coarse_prev=None, depth=0):
    global function_evaluations

    c = (a + b) / 2

    # Hitung integral 
    if I_coarse_prev is None:
        I_coarse = calculate_I_coarse(func, a, b)
        function_evaluations += 3
    else:
        I_coarse = I_coarse_prev

    # Hitung integral halus (dua sub-interval)
    I_left = calculate_I_coarse(func, a, c)
    I_right = calculate_I_coarse(func, c, b)
    I_fine = I_left + I_right

    function_evaluations += 2

    # Estimasi error
    error_est = abs(I_fine - I_coarse) / 15

    # Catat untuk grafik
    errors.append(error_est)
    interval_sizes.append(b - a)

    # Format log
    indent = "  " * depth
    status = " ACCEPTED " if error_est < tol else " SUBDIVIDE"

    print(
        f"{depth:<5d} | "
        f"{indent}[{a:7.5f}, {b:7.5f}]".ljust(32) + " | "
        f"h = {(b-a):9.5f} | "
        f"Err = {error_est:12.3e} | "
        f"Tol = {tol:12.3e} | "
        f"{status}"
    )

    # Keputusan
    if error_est < tol:
        intervals_final.append((a, b))
        return I_fine

    # Subdivide dua sisi
    new_tol = tol / 2
    L = adaptive_integration(func, a, c, new_tol, I_left, depth + 1)
    R = adaptive_integration(func, c, b, new_tol, I_right, depth + 1)

    return L + R


A = 0
B = np.pi / 2
TOL = 1e-6

print("\n" + "="*112)
print("DEPTH | INTERVAL                  |     h      |   ERROR EST         |    TOLERANCE      | STATUS")
print("="*112)

result = adaptive_integration(f, A, B, TOL)
exact = 1.0  # sin(pi/2)

print("\n" + "="*112)
print("                         HASIL AKHIR")
print("="*112)
print(f"Hasil Integral Adaptif : {result:.12f}")
print(f"Hasil Eksak            : {exact:.12f}")
print(f"Error Akhir            : {abs(exact - result):.12e}")
print(f"Total Evaluasi Fungsi  : {function_evaluations}")
print(f"Jumlah Interval Akhir  : {len(intervals_final)}")
print("="*112 + "\n")

# Error per iterasi (log scale)
plt.figure(figsize=(10,5))
plt.plot(errors, linewidth=2)
plt.yscale("log")
plt.xlabel("Iterasi")
plt.ylabel("Error Estimasi (log scale)")
plt.title("Perubahan Error Simpson Adaptif per Iterasi")
plt.grid(True)
plt.show()

# Error vs Ukuran Interval
plt.figure(figsize=(10,5))
plt.scatter(interval_sizes, errors)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ukuran Interval")
plt.ylabel("Error")
plt.title("Error vs Ukuran Interval")
plt.grid(True)
plt.show()

# Rekonstruksi fungsi + interval adaptif
plt.figure(figsize=(12,4))
xs = np.linspace(A, B, 400)
plt.plot(xs, f(xs), label="cos(x)", linewidth=2)

for (a, b) in intervals_final:
    plt.axvspan(a, b, alpha=0.25)

plt.title("Rekonstruksi Fungsi + Interval Adaptif")
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.grid(True)
plt.show()
