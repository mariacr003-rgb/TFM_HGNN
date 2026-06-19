#src/01_validate_data.py

import csv, sys
from pathlib import Path

EXPECTED_CLINICAL = ["case_id", "vital_status", "days_to_death",
                      "days_to_last_follow_up", "age_at_diagnosis",
                      "ajcc_pathologic_stage"]


def validar_clinical(in_path, out_valid, out_report_lines):
    rows_validas = []
    n_total = 0
    n_descartados = 0

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != EXPECTED_CLINICAL:
            out_report_lines.append(f"ERROR\tBadHeader:{reader.fieldnames}")
            return None

        for row in reader:
            n_total += 1
            tiene_muerte = row["days_to_death"].strip() != ""
            tiene_seguimiento = row["days_to_last_follow_up"].strip() != ""
            if not row["vital_status"].strip() or not (tiene_muerte or tiene_seguimiento):
                n_descartados += 1
                continue
            rows_validas.append(row)

    out_valid.parent.mkdir(parents=True, exist_ok=True)
    with out_valid.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPECTED_CLINICAL, delimiter="\t")
        w.writeheader()
        w.writerows(rows_validas)

    out_report_lines.append(
        f"Clinical: {n_total} pacientes -> {len(rows_validas)} validos "
        f"({n_descartados} descartados por falta de seguimiento)"
    )
    return rows_validas


def validar_rnaseq(in_path, out_valid, out_report_lines):
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        n_muestras = len(header) - 1
        filas = list(reader)

    n_genes_inicial = len(filas)
    filas_validas = []

    for fila in filas:
        valores = fila[1:]
        n_vacios = sum(1 for v in valores if v.strip() == "")
        pct_vacios = n_vacios / n_muestras
        if pct_vacios <= 0.5:
            filas_validas.append(fila)

    out_valid.parent.mkdir(parents=True, exist_ok=True)
    with out_valid.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(filas_validas)

    out_report_lines.append(
        f"RNA-seq: {n_genes_inicial} genes, {n_muestras} muestras -> "
        f"{len(filas_validas)} genes tras filtrar >50% de valores vacios"
    )
    return filas_validas


def main(cohorte, in_clinical, in_rnaseq, out_dir):
    in_clinical = Path(in_clinical)
    in_rnaseq = Path(in_rnaseq)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reporte = []

    out_clinical_valid = out_dir / f"{cohorte}_clinical_valid.tsv"
    out_rnaseq_valid = out_dir / f"{cohorte}_rnaseq_valid.tsv"
    out_report = out_dir / f"{cohorte}_validation_report.txt"

    clin_validas = validar_clinical(in_clinical, out_clinical_valid, reporte)
    if clin_validas is None:
        out_report.write_text("\n".join(reporte), encoding="utf-8")
        print(f"[{cohorte}] VALIDACION FALLIDA. Ver informe.")
        return 2

    validar_rnaseq(in_rnaseq, out_rnaseq_valid, reporte)

    out_report.write_text("\n".join(reporte), encoding="utf-8")

    print(f"[{cohorte}] Validacion completada.")
    print(f"  -> {out_clinical_valid}")
    print(f"  -> {out_rnaseq_valid}")
    print(f"  -> {out_report}")
    return 0


if __name__ == "__main__":
    cohorte, in_clinical, in_rnaseq, out_dir = sys.argv[1:5]
    sys.exit(main(cohorte, in_clinical, in_rnaseq, out_dir))