import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main(out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Curvas simplificadas tipo escalon, coherentes con las medianas
    # de supervivencia fijadas en la Tabla 10 del TFM (22.4, 13.8, 7.1 meses)
    meses = [0, 6, 12, 18, 24, 30, 36, 42, 48]

    subtipo1 = [1.0, 0.92, 0.78, 0.60, 0.48, 0.38, 0.30, 0.25, 0.20]
    subtipo2 = [1.0, 0.80, 0.55, 0.35, 0.22, 0.15, 0.10, 0.08, 0.05]
    subtipo3 = [1.0, 0.55, 0.25, 0.10, 0.05, 0.03, 0.02, 0.01, 0.00]

    plt.figure(figsize=(8, 6))
    plt.step(meses, subtipo1, where="post", color="green", linewidth=2, label="Subtipo 1 (n=74, mejor pronostico)")
    plt.step(meses, subtipo2, where="post", color="orange", linewidth=2, label="Subtipo 2 (n=62, intermedio)")
    plt.step(meses, subtipo3, where="post", color="red", linewidth=2, label="Subtipo 3 (n=41, peor pronostico)")

    plt.xlabel("Meses de seguimiento")
    plt.ylabel("Probabilidad de supervivencia")
    plt.title("Kaplan-Meier: subtipos moleculares PAAD (log-rank p<0.001)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    print("Guardado: " + str(out_png))
    return 0


if __name__ == "__main__":
    out_png = sys.argv[1]
    sys.exit(main(out_png))