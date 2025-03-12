import numpy as np
import matplotlib.pyplot as plt

# Implementa y ejecuta el código para resolver cada sistema con el método de Jacobi.
# Incluye en tu informe:
# 1. Tablas con las soluciones iterativas.
# 2. Gráficas de los errores absoluto, relativo y cuadrático.
# 3. Análisis de la convergencia en cada caso.
# 4. Comparación con la solución exacta usando numpy.linalg.solve(A, b).

# Definir el nuevo sistema de ecuaciones
A = np.array([[8, 2, -1, 0, 0, 0],
              [3, 15, -2, 1, 0, 0],
              [0, -2, 12, 2, -1, 0],
              [0, 1, -1, 9, -2, 1],
              [0, 0, -2, 3, 14, 1],
              [0, 0, 0, 1, -2, 10]])

b = np.array([10, 24, -18, 16, -9, 22])

# Solución exacta para comparar errores
sol_exacta = np.linalg.solve(A, b)

# Criterio de paro
tolerancia = 1e-6
max_iter = 100

def jacobi(A, b, tol, max_iter):
    n = len(A)
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    soluciones_iter = []
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Guardar la solución iterativa
        soluciones_iter.append(x_new.copy())
        
        # Calcular errores
        error_abs = np.linalg.norm(x_new - sol_exacta, ord=1)
        error_rel = np.linalg.norm(x_new - sol_exacta, ord=1) / np.linalg.norm(sol_exacta, ord=1)
        error_cuad = np.linalg.norm(x_new - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        # Criterio de convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new
    
    return x, errores_abs, errores_rel, errores_cuad, soluciones_iter, k+1

# Ejecutar el método de Jacobi
sol_aprox, errores_abs, errores_rel, errores_cuad, soluciones_iter, iteraciones = jacobi(A, b, tolerancia, max_iter)

# Mostrar tabla de errores por iteración
print("\nTabla de errores por iteración:")
print("Iteración | Error absoluto | Error relativo | Error cuadrático")
print("-------------------------------------------------------------")
for i in range(iteraciones):
    print(f"{i+1:^9} | {errores_abs[i]:^14.6f} | {errores_rel[i]:^14.6f} | {errores_cuad[i]:^14.6f}")

# Guardar y mostrar la gráfica de errores
plt.figure(figsize=(8,6))
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='d')
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Convergencia de los errores en el método de Jacobi")
plt.legend()
plt.grid()
plt.savefig("errores_jacobi.png")  # Guardar la figura en archivo PNG
plt.show()  # Mostrar la gráfica en pantalla

# Mostrar tabla de comparación entre solución aproximada y exacta
print("\nTabla de comparación entre solución aproximada y exacta:")
print("Variable | Solución aproximada | Solución exacta | Diferencia absoluta")
print("--------------------------------------------------------------------")
for i in range(len(sol_aprox)):
    print(f"x{i+1:^6} | {sol_aprox[i]:^19.6f} | {sol_exacta[i]:^15.6f} | {np.abs(sol_aprox[i] - sol_exacta[i]):^20.6f}")

# Análisis de convergencia
print("\nAnálisis de la convergencia:")
if iteraciones < max_iter:
    print(f"El método de Jacobi convergió en {iteraciones} iteraciones.")
else:
    print("El método de Jacobi no convergió dentro del número máximo de iteraciones.")
