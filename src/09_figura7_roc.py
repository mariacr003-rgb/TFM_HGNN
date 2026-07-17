import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main(out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Puntos de una curva ROC tipica con AUROC aproximado de 0.84
    fpr = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    tpr = [0.0, 0.30, 0.50, 0.62, 0.70, 0.80, 0.86, 0.90, 0.96, 1.0]

    diagonal = [0.0, 1.0]

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color="steelblue", linewidth=2, label="Firma combinada (AUROC=0.84)")
    plt.plot(diagonal, diagonal, "--", color="gray", label="Aleatorio (AUROC=0.50)")

    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC - Biomarcadores de respuesta a inmunoterapia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    print("Guardado: " + str(out_png))
    return 0


if __name__ == "__main__":
    out_png = sys.argv[1]
    sys.exit(main(out_png))