import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

# Calcula estadisticos descriptivos del tiempo de seguimiento (mediana,
# media, cuartiles, minimo, maximo) por cohorte, para la Tabla 4 y la
# Seccion 3.3 del TFM. Ninguno de los dos scripts de la Fase 1
# relacionados (05_estadisticas_descriptivas.py, guarda solo total/
# fallecidos/vivos/edad/estadio; 06_grafico_supervivencia.py, solo
# imprime por pantalla el recuento de pacientes validos, sin guardar
# mediana/rango en ningun fichero) calculaba ni guardaba estos valores
# - de ahi este script nuevo, en vez de dejarlo como un calculo suelto
# no reproducible.
#
# Entrada: la misma tabla combinada de las 5 cohortes que ya alimenta
# los histogramas de supervivencia (ver
# src/33_exportar_datos_supervivencia.py), NO se descarga ni recalcula
# ningun dato nuevo desde el GDC. tiempo_dias se convierte a meses
# (dias/30) para que las unidades coincidan con el eje X de los
# histogramas ("Meses de seguimiento",
# src/06_grafico_supervivencia.py).

COHORTES = ["BRCA", "LUAD", "LUSC", "COAD", "KIRC"]


def cargar_datos(ruta_entrada):
    datos = defaultdict(list)
    with open(ruta_entrada, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            datos[row["cohorte"]].append((float(row["tiempo_dias"]) / 30, row["vital_status"]))
    return datos


def main(ruta_entrada, ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    datos = cargar_datos(ruta_entrada)

    filas_salida = []
    for cohorte in COHORTES:
        registros = datos[cohorte]
        meses = np.array([m for m, _ in registros])
        n = len(registros)
        n_dead = sum(1 for _, v in registros if v == "Dead")
        n_alive = sum(1 for _, v in registros if v == "Alive")
        q1, mediana, q3 = np.percentile(meses, [25, 50, 75])

        fila = {
            "cohorte": cohorte,
            "n_valido": n,
            "n_fallecidos": n_dead,
            "pct_fallecidos": round(100 * n_dead / n, 1),
            "n_vivos": n_alive,
            "pct_vivos": round(100 * n_alive / n, 1),
            "mediana_meses": round(float(mediana), 2),
            "media_meses": round(float(np.mean(meses)), 2),
            "q1_meses": round(float(q1), 2),
            "q3_meses": round(float(q3), 2),
            "min_meses": round(float(meses.min()), 2),
            "max_meses": round(float(meses.max()), 2),
        }
        filas_salida.append(fila)
        print(f"{cohorte}: n={fila['n_valido']}, mediana={fila['mediana_meses']}, "
              f"media={fila['media_meses']}, Q1={fila['q1_meses']}, Q3={fila['q3_meses']}, "
              f"min={fila['min_meses']}, max={fila['max_meses']}")

    campos = ["cohorte", "n_valido", "n_fallecidos", "pct_fallecidos", "n_vivos", "pct_vivos",
              "mediana_meses", "media_meses", "q1_meses", "q3_meses", "min_meses", "max_meses"]
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, delimiter="\t")
        writer.writeheader()
        writer.writerows(filas_salida)

    print(f"Guardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    import sys
    ruta_entrada = sys.argv[1] if len(sys.argv) > 1 else "results/2026-08-20-paso54/datos_supervivencia_5_cohortes.tsv"
    ruta_salida = sys.argv[2] if len(sys.argv) > 2 else "results/2026-08-20-paso54/estadisticas_seguimiento_5_cohortes.tsv"
    sys.exit(main(ruta_entrada, ruta_salida))
