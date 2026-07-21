"""
Pregunta 7 - Matriz inversa mediante Gauss-Jordan.

La función inv_matrix se basa en la función gauss_jordan del código
entregado por el profesor. No se usa numpy.linalg.inv ni scipy.
Cada columna de A^-1 se obtiene resolviendo A*x=e_j.
"""

from typing import Sequence


try:
    from src import matriz_aumentada as matriz_aumentada_base
    from src import gauss_jordan as gauss_jordan_base
except ImportError:
    matriz_aumentada_base = None
    gauss_jordan_base = None


def matriz_aumentada_local(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
) -> list[list[float]]:
    return [
        [float(valor) for valor in fila] + [float(b[i])]
        for i, fila in enumerate(A)
    ]


def gauss_jordan_local(Ab: Sequence[Sequence[float]]) -> list[float]:
    m = [list(map(float, fila)) for fila in Ab]
    n = len(m)

    if any(len(fila) != n + 1 for fila in m):
        raise ValueError("La matriz aumentada debe tener n filas y n+1 columnas.")

    for columna in range(n):
        pivote = max(range(columna, n), key=lambda i: abs(m[i][columna]))
        if abs(m[pivote][columna]) < 1e-14:
            raise ValueError("La matriz no es invertible.")

        m[columna], m[pivote] = m[pivote], m[columna]
        divisor = m[columna][columna]
        m[columna] = [valor / divisor for valor in m[columna]]

        for fila in range(n):
            if fila == columna:
                continue
            factor = m[fila][columna]
            m[fila] = [
                m[fila][j] - factor * m[columna][j]
                for j in range(n + 1)
            ]

    return [m[i][-1] for i in range(n)]


def inv_matrix(A: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(A)
    if n == 0 or any(len(fila) != n for fila in A):
        raise ValueError("A debe ser una matriz cuadrada no vacía.")

    matriz_aumentada = matriz_aumentada_base or matriz_aumentada_local
    gauss_jordan = gauss_jordan_base or gauss_jordan_local

    inversa = [[0.0 for _ in range(n)] for _ in range(n)]

    for columna in range(n):
        e = [0.0] * n
        e[columna] = 1.0
        Ab = matriz_aumentada(A, e)
        solucion = gauss_jordan(Ab)

        for fila in range(n):
            inversa[fila][columna] = float(solucion[fila])

    return inversa


def producto(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(len(B)))
            for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]


def error_identidad(A: list[list[float]], A_inv: list[list[float]]) -> float:
    producto_resultado = producto(A, A_inv)
    n = len(A)
    return max(
        abs(producto_resultado[i][j] - (1.0 if i == j else 0.0))
        for i in range(n)
        for j in range(n)
    )


def imprimir_matriz(matriz: list[list[float]]) -> None:
    for fila in matriz:
        valores = []
        for valor in fila:
            redondeado = round(valor)
            if abs(valor - redondeado) < 1e-10:
                valores.append(f"{redondeado:8d}")
            else:
                valores.append(f"{valor:12.6f}")
        print("[ " + " ".join(valores) + " ]")


EJERCICIOS = {
    "Ejercicio 1": [[2, -3], [-1, 1]],
    "Ejercicio 2": [
        [4, 0, 0, 5],
        [1, 0, 4, 0],
        [3, 4, 1, 3],
        [1, 3, 3, 0],
    ],
    "Ejercicio 3": [
        [0, 0, 0, 0, 0, 0, 1, -1],
        [0, 1, -1, 1, 0, -1, 0, 1],
        [-1, -1, 0, 0, 2, 1, 0, 0],
        [-1, -1, -1, 1, 2, 0, 0, 1],
        [-1, 1, 1, 0, -1, -1, 0, 2],
        [0, 1, 0, 0, -1, -1, 0, 0],
        [1, -1, -1, 1, 2, 1, 0, 2],
        [2, 0, 0, 0, 0, 1, 2, 0],
    ],
    "Ejercicio 4": [
        [1, 0, 0, 0, -1, 0, 0, -1, 1, -1],
        [1, 1, 0, -1, -1, 1, 0, 0, 1, -1],
        [-1, 0, -1, 0, 0, 0, -1, 1, 0, 0],
        [0, 0, -1, 0, -1, -1, 1, 0, 1, 0],
        [-1, 0, 0, -1, 1, 1, 1, 1, 0, -1],
        [1, 0, 0, 1, -1, -1, -1, 1, -1, 0],
        [1, 1, 1, 0, 1, 0, -1, -1, -1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, 1, 1, 1, -1],
    ],
}


def main() -> None:
    for nombre, A in EJERCICIOS.items():
        print(f"\n{nombre}")
        print("-" * len(nombre))
        inversa = inv_matrix(A)
        imprimir_matriz(inversa)
        print(f"Error máximo de A*A^-1-I: {error_identidad(A, inversa):.3e}")


if __name__ == "__main__":
    main()
