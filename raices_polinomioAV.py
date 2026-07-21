from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect, newton



def f(x: float) -> float:
    return (
        x**5
        + 5.25 * x**4
        + 4.125 * x**3
        - 9.125 * x**2
        - 14.625 * x
        - 5.625
    )


def df(x: float) -> float:
    return (
        5 * x**4
        + 21 * x**3
        + 12.375 * x**2
        - 18.25 * x
        - 14.625
    )


def d2f(x: float) -> float:
    return 20 * x**3 + 63 * x**2 + 24.75 * x - 18.25


def agregar_raiz_unica(raices: list[float], candidata: float, tol: float = 1e-5) -> None:
    if not math.isfinite(candidata):
        return
    if abs(f(candidata)) > 1e-7:
        return
    if not any(abs(candidata - existente) < tol for existente in raices):
        raices.append(candidata)


def encontrar_raices() -> list[float]:
    raices: list[float] = []

    # Bisección: encuentra las raíces de multiplicidad impar.
    malla = np.linspace(-10, 10, 4001)
    for izquierda, derecha in zip(malla[:-1], malla[1:]):
        fi = f(float(izquierda))
        fd = f(float(derecha))

        if abs(fi) < 1e-12:
            agregar_raiz_unica(raices, float(izquierda))
        elif fi * fd < 0:
            raiz = bisect(f, float(izquierda), float(derecha), xtol=1e-13)
            agregar_raiz_unica(raices, raiz)

    # Para reforzar la búsqueda de raíces múltiples, se buscan puntos
    # críticos resolviendo f'(x)=0 con Newton y se comprueba cuáles también
    # satisfacen f(x)=0.
    puntos_criticos: list[float] = []
    for semilla in np.linspace(-6, 4, 161):
        try:
            critico = newton(
                df,
                float(semilla),
                fprime=d2f,
                tol=1e-13,
                maxiter=200,
            )
        except (RuntimeError, OverflowError, ZeroDivisionError):
            continue

        if (
            math.isfinite(critico)
            and not any(abs(critico - previo) < 1e-6 for previo in puntos_criticos)
        ):
            puntos_criticos.append(critico)

    for critico in puntos_criticos:
        if abs(f(critico)) < 1e-8:
            agregar_raiz_unica(raices, critico)

    # Redondear elimina el ruido numérico de Newton, por ejemplo
    # -0.999999994467... se presenta como -1.
    return sorted(round(raiz, 6) for raiz in raices)


def graficar(raices: list[float]) -> Path:
    salida = Path(__file__).resolve().parent / "resultados" / "pregunta_4_raices.png"
    salida.parent.mkdir(exist_ok=True)

    xs = np.linspace(-5, 3, 1500)
    ys = [f(float(x)) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, linewidth=1)
    plt.scatter(raices, [0] * len(raices), marker="o", label="Raíces")
    plt.ylim(-80, 80)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Raíces del polinomio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida, dpi=180)
    plt.close()
    return salida


def main() -> None:
    raices = encontrar_raices()
    respuestas = raices + [math.nan] * (6 - len(raices))

    print("RAÍCES REALES, SIN REPETIR Y EN ORDEN ASCENDENTE")
    for indice, raiz in enumerate(respuestas[:6], start=1):
        if math.isnan(raiz):
            print(f"x{indice}=NaN")
        else:
            print(f"x{indice}={raiz:g}")

    print("\nVerificación:")
    for raiz in raices:
        print(f"f({raiz:g})={f(raiz):.12e}")

    grafica = graficar(raices)
    print(f"\nGráfica guardada en: {grafica}")


if __name__ == "__main__":
    main()
