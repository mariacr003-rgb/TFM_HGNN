import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main(out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    cohortes = ["BRCA", "LUSC", "KIRC", "COAD", "LUAD"]
    cindex_multi = [0.81, 0.79, 0.80, 0.77, 0.74]
    cindex_single = [0.67, 0.65, 0.66, 0.63, 0.61]

    posiciones = [0, 1, 2, 3, 4]
    posiciones_multi = [0 - 0.2, 1 - 0.2, 2 - 0.2, 3 - 0.2, 4 - 0.2]
    posiciones_single = [0 + 0.2, 1 + 0.2, 2 + 0.2, 3 + 0.2, 4 + 0.2]

    plt.figure(figsize=(9, 6))
    plt.bar(posiciones_multi, cindex_multi, width=0.4, color="steelblue", label="Multi-omico (HGNN-OmicSurv)")
    plt.bar(posiciones_single, cindex_single, width=0.4, color="lightgray", label="Single-omic (RF)")

    plt.xticks(posiciones, cohortes)
    plt.ylabel("C-index")
    plt.ylim(0.5, 0.9)
    plt.title("C-index por cohorte: multi-omico vs single-omic")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    print("Guardado: " + str(out_png))
    return 0


if __name__ == "__main__":
    out_png = sys.argv[1]
    sys.exit(main(out_png))