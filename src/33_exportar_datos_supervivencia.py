import csv, sys
from pathlib import Path

# Exporta, para las 5 cohortes, exactamente los mismos datos que
# alimentan los histogramas de src/06_grafico_supervivencia.py (Paso
# 54) - mismo filtro replicado aqui, sin duplicar el resto del script:
# solo pacientes con vital_status Dead+days_to_death valido o Alive+
# days_to_last_follow_up valido. Pacientes sin ese dato, o con
# vital_status distinto de Alive/Dead, quedan excluidos (igual que en
# el histograma).
#
# tiempo_dias es el valor CRUDO del campo del GDC (days_to_death o
# days_to_last_follow_up segun corresponda), SIN convertir a meses -
# la conversion a meses (dias/30) es una decision de presentacion del
# histograma, no del dato en si.

COHORTES = ["BRCA", "LUAD", "LUSC", "COAD", "KIRC"]


def extraer_cohorte(cohorte):
    ruta = Path("data/raw") / f"{cohorte}_clinical.tsv"
    filas = []
    with ruta.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vital = row["vital_status"]
            if vital == "Dead" and row["days_to_death"] != "":
                filas.append((cohorte, row["days_to_death"], "Dead"))
            elif vital == "Alive" and row["days_to_last_follow_up"] != "":
                filas.append((cohorte, row["days_to_last_follow_up"], "Alive"))
    return filas


def main(ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    todas_las_filas = []
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["cohorte", "tiempo_dias", "vital_status"])
        for cohorte in COHORTES:
            filas = extraer_cohorte(cohorte)
            todas_las_filas.extend(filas)
            writer.writerows(filas)
            n_dead = sum(1 for _, _, v in filas if v == "Dead")
            n_alive = sum(1 for _, _, v in filas if v == "Alive")
            print(f"  {cohorte}: {len(filas)} filas ({n_dead} Dead, {n_alive} Alive)")

    print(f"Total: {len(todas_las_filas)} filas")
    print(f"Guardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    ruta_salida = sys.argv[1] if len(sys.argv) > 1 else "results/2026-08-20-paso54/datos_supervivencia_5_cohortes.tsv"
    sys.exit(main(ruta_salida))
