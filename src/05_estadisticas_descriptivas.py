import csv, sys
from pathlib import Path


def main(in_clinical, out_path):
    in_clinical = Path(in_clinical)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_dead = 0
    n_alive = 0
    suma_edades = 0
    n_edades = 0

    estadios_nombres = []
    estadios_conteos = []

    with in_clinical.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_total += 1

            if row["vital_status"] == "Dead":
                n_dead += 1
            if row["vital_status"] == "Alive":
                n_alive += 1

            edad_dias = row["age_at_diagnosis"]
            if edad_dias != "":
                edad_anos = int(edad_dias) / 365
                suma_edades += edad_anos
                n_edades += 1

            estadio = row["ajcc_pathologic_stage"]
            if estadio != "":
                if estadio in estadios_nombres:
                    indice = estadios_nombres.index(estadio)
                    estadios_conteos[indice] += 1
                else:
                    estadios_nombres.append(estadio)
                    estadios_conteos.append(1)

    edad_media = suma_edades / n_edades

    lineas = []
    lineas.append("Total de pacientes: " + str(n_total))
    lineas.append("Fallecidos (Dead): " + str(n_dead))
    lineas.append("Vivos (Alive): " + str(n_alive))
    lineas.append("Edad media al diagnostico: " + str(round(edad_media, 1)) + " anos")
    lineas.append("")
    lineas.append("Distribucion por estadio AJCC:")

    for i in range(len(estadios_nombres)):
        lineas.append("  " + estadios_nombres[i] + ": " + str(estadios_conteos[i]))

    texto_final = "\n".join(lineas)
    out_path.write_text(texto_final, encoding="utf-8")

    print(texto_final)
    return 0


if __name__ == "__main__":
    in_clinical, out_path = sys.argv[1:3]
    sys.exit(main(in_clinical, out_path))