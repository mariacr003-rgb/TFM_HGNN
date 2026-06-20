import csv, sys
import matplotlib.pyplot as plt
from pathlib import Path


def main(in_clinical, out_png):
    in_clinical = Path(in_clinical)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    meses_dead = []
    meses_alive = []

    with in_clinical.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vital = row["vital_status"]
            dias_muerte = row["days_to_death"]
            dias_seguimiento = row["days_to_last_follow_up"]

            if vital == "Dead" and dias_muerte != "":
                meses = float(dias_muerte) / 30
                meses_dead.append(meses)

            if vital == "Alive" and dias_seguimiento != "":
                meses = float(dias_seguimiento) / 30
                meses_alive.append(meses)

    print("Pacientes fallecidos con datos validos: " + str(len(meses_dead)))
    print("Pacientes vivos con datos validos: " + str(len(meses_alive)))

    plt.figure(figsize=(8, 6))
    plt.hist(meses_alive, bins=20, alpha=0.6, color="steelblue", label="Vivos (Alive)")
    plt.hist(meses_dead, bins=20, alpha=0.6, color="indianred", label="Fallecidos (Dead)")
    plt.xlabel("Meses de seguimiento")
    plt.ylabel("Numero de pacientes")
    plt.title("Distribucion de tiempos de seguimiento - TCGA-BRCA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")

    print("Guardado: " + str(out_png))
    return 0


if __name__ == "__main__":
    in_clinical, out_png = sys.argv[1:3]
    sys.exit(main(in_clinical, out_png))