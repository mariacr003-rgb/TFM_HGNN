import csv, sys, gzip, tarfile, io
from pathlib import Path

HEADER_SALIDA = ["case_id", "vital_status", "days_to_death",
                  "days_to_last_follow_up", "age_at_diagnosis",
                  "ajcc_pathologic_stage"]

# Nombre del fichero clinico dentro del paquete tar.gz que descarga el
# boton "Clinical" del portal GDC cuando se guarda sin extraer a mano
NOMBRE_MIEMBRO_TAR = "clinical.tsv"


def abrir_clinical(in_path):
    # El GDC puede entregar el fichero clinico de 3 formas: .tsv plano,
    # .tsv.gz (gzip de un unico fichero), o un paquete tar.gz (aunque
    # tenga extension .tsv.gz) con varias tablas dentro
    # (clinical.tsv, exposure.tsv, family_history.tsv, follow_up.tsv,
    # pathology_detail.tsv); en ese caso usamos solo clinical.tsv.
    if tarfile.is_tarfile(in_path):
        tar = tarfile.open(in_path, "r:*")
        miembro = tar.extractfile(NOMBRE_MIEMBRO_TAR)
        return io.TextIOWrapper(miembro, encoding="utf-8", newline="")
    if in_path.suffix == ".gz":
        return gzip.open(in_path, "rt", encoding="utf-8", newline="")
    return in_path.open("r", encoding="utf-8", newline="")


def main(in_path, out_path):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filas_salida = []
    vistos = []

    with abrir_clinical(in_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            case_id = row["cases.submitter_id"].strip()
            if not case_id or case_id in vistos:
                continue
            vistos.append(case_id)

            vital_status = row["demographic.vital_status"].strip()
            days_death = row["demographic.days_to_death"].strip()
            days_followup = row["diagnoses.days_to_last_follow_up"].strip()
            age = row["diagnoses.age_at_diagnosis"].strip()
            stage = row["diagnoses.ajcc_pathologic_stage"].strip()

            # El GDC usa "'--" para indicar que el valor no esta disponible
            if days_death == "'--":
                days_death = ""
            if days_followup == "'--":
                days_followup = ""
            if age == "'--":
                age = ""
            if stage == "'--":
                stage = ""

            filas_salida.append([case_id, vital_status, days_death,
                                  days_followup, age, stage])

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(HEADER_SALIDA)
        w.writerows(filas_salida)

    print(f"Convertido: {out_path} ({len(filas_salida)} pacientes unicos)")
    return 0


if __name__ == "__main__":
    in_path, out_path = sys.argv[1:3]
    sys.exit(main(in_path, out_path))