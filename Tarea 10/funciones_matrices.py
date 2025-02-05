import numpy as np

def es_no_singular(matriz):
    """
    Determina si una matriz es no singular (tiene inversa).
    """
    # Calcula el determinante de la matriz
    determinante = np.linalg.det(matriz)
    
    # Si el determinante es distinto de cero, es no singular
    return not np.isclose(determinante, 0)

def calcular_inversa(matriz):
    """
    Calcula la inversa de una matriz si es no singular.
    """
    if es_no_singular(matriz):
        inversa = np.linalg.inv(matriz)
        return inversa
    else:
        raise ValueError("La matriz es singular y no tiene inversa.")
#################################################################################################################

import numpy as np    
def resolver_sistema_4x4(A, b):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b, donde A es una matriz 4x4 y b es un vector de tamaño 4.
    
    Parámetros:
        A (numpy.ndarray): Matriz de coeficientes de tamaño 4x4.
        b (numpy.ndarray): Vector de términos independientes de tamaño 4.
    
    Retorna:
        x (numpy.ndarray): Solución del sistema de ecuaciones.
    """
    # Verificar que A sea una matriz 4x4 y b sea un vector de tamaño 4
    if A.shape != (4, 4) or b.shape != (4,):
        raise ValueError("La matriz A debe ser 4x4 y el vector b debe ser de tamaño 4.")
    
    # Resolver el sistema de ecuaciones
    x = np.linalg.solve(A, b)
    return x
#################################################################################################################

def eliminacion_gaussiana_lu(A: np.ndarray):
    """Realiza la descomposición LU de una matriz A usando eliminación gaussiana.

    ## Parameters

    ``A``: matriz de coeficientes del sistema de ecuaciones lineales. Debe ser de tamaño n-by-n.

    ## Return

    ``L``: matriz triangular inferior con unos en la diagonal.
    ``U``: matriz triangular superior.
    """
    if not isinstance(A, np.ndarray):
        print("Convirtiendo A a numpy array.")
        A = np.array(A)
    assert A.shape[0] == A.shape[1], "La matriz A debe ser cuadrada."
    n = A.shape[0]

    L = np.eye(n)  # Matriz identidad (L tendrá unos en la diagonal)
    U = A.copy()   # Copia de A para convertirse en U

    for i in range(n - 1):  # loop por columna
        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if U[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(U[pi, i]) < abs(U[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            raise ValueError("No existe solución única.")

        if p != i:
            # swap rows
            print(f"Intercambiando filas {i} y {p}")
            _aux = U[i, :].copy()
            U[i, :] = U[p, :].copy()
            U[p, :] = _aux

            # También intercambiar las filas correspondientes en L
            _aux = L[i, :i].copy()
            L[i, :i] = L[p, :i].copy()
            L[p, :i] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = U[j, i] / U[i, i]
            L[j, i] = m  # Guardar el multiplicador en L
            U[j, i:] = U[j, i:] - m * U[i, i:]

        print(f"\nL:\n{L}\nU:\n{U}")

    return L, U

def sustitucion_adelante(L, b):
    """Resuelve el sistema Ly = b usando sustitución hacia adelante.

    ## Parameters

    ``L``: matriz triangular inferior con unos en la diagonal.
    ``b``: vector de términos independientes.

    ## Return

    ``y``: vector solución.
    """
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        suma = np.dot(L[i, :i], y[:i])
        y[i] = (b[i] - suma) / L[i, i]

    return y

def sustitucion_atras(U, y):
    """Resuelve el sistema Ux = y usando sustitución hacia atrás.

    ## Parameters

    ``U``: matriz triangular superior.
    ``y``: vector solución de Ly = b.

    ## Return

    ``x``: vector solución.
    """
    n = U.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        suma = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (y[i] - suma) / U[i, i]

    return x

def resolver_sistema_lu(A, b):
    """Resuelve el sistema Ax = b usando la descomposición LU.

    ## Parameters

    ``A``: matriz de coeficientes del sistema de ecuaciones lineales.
    ``b``: vector de términos independientes.

    ## Return

    ``x``: vector solución.
    """
    L, U = eliminacion_gaussiana_lu(A)
    y = sustitucion_adelante(L, b)
    x = sustitucion_atras(U, y)
    return x
