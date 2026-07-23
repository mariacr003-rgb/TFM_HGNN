import csv, sys
from pathlib import Path

PATRON_BUSCADO = "methylation_array.sesame.level3betas.txt"


def main(in_manifest, out_manifest, n_max, size_min=None, size_max=None):
    in_manifest = Path(in_manifest)
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    with in_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        filas_filtradas = []
        for fila in reader:
            filename = fila[1]
            if PATRON_BUSCADO not in filename:
                continue
            # size_min/size_max permiten quedarse solo con un array
            # concreto (ej. 450K), usando el tamano en bytes del propio
            # manifest, que coincide exacto con el tamano real del
            # fichero descargado
            size = int(fila[3])
            if size_min is not None and size < size_min:
                continue
            if size_max is not None and size > size_max:
                continue
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
    size_min = int(sys.argv[4]) if len(sys.argv) > 4 else None
    size_max = int(sys.argv[5]) if len(sys.argv) > 5 else None
    sys.exit(main(in_manifest, out_manifest, n_max, size_min, size_max))