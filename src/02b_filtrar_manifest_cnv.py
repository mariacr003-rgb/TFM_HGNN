import csv, sys
from pathlib import Path

# El patron "gene_level_copy_number.v36.tsv" corresponde al sufijo fijo  de los ficheros de CNV generados por el pipeline ASCAT2 del GDC
#  Verificado directamente en el manifest completo del proyecto TCGA-BRCA (data/raw/brca_manifest_full.txt),
# descargado desde https://portal.gdc.cancer.gov/projects/TCGA-BRCA

PATRON_BUSCADO = "gene_level_copy_number.v36.tsv"

def main(in_manifest, out_manifest, n_max):
    in_manifest = Path(in_manifest)
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with in_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        filas_filtradas = []
        for fila in reader:
            filename = fila[1]
            if PATRON_BUSCADO in filename:
                filas_filtradas.append(fila)
                if len(filas_filtradas) >= n_max:
                    break

    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(filas_filtradas)

    print(f"Filtrado: {len(filas_filtradas)} ficheros de RNA-seq")
    print(f"Guardado en: {out_manifest}")
    return 0


if __name__ == "__main__":
    in_manifest, out_manifest, n_max = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.exit(main(in_manifest, out_manifest, n_max))