import csv, sys
from pathlib import Path


def leer_fichero_paciente(ruta):
    # La cabecera del fichero crudo (gene_id, gene_name, chromosome,
    # start, end, copy_number, min_copy_number, max_copy_number) tiene
    # las mismas 8 columnas que las filas de datos, asi que el filtro
    # "len(fila) < 8" no la distinguia: se colaba como una fila de gen
    # fantasma (gene_id="gene_id", copy_number="copy_number"). Se salta
    # explicitamente con next(reader) (hallazgo y correccion del Paso 32,
    # ver results/2026-08-01-paso32/runlog.txt).
    genes = []
    valores = []
    with ruta.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for fila in reader:
            if len(fila) < 8:
                continue
            gene_id = fila[0]
            copy_number = fila[5]
            genes.append(gene_id)
            valores.append(copy_number)
    return genes, valores


def main(in_dir, out_path):
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    carpetas = [d for d in in_dir.iterdir() if d.is_dir()]

    primera_carpeta = carpetas[0]
    fichero_ref = list(primera_carpeta.glob("*gene_level_copy_number.v36.tsv"))[0]
    genes_referencia, _ = leer_fichero_paciente(fichero_ref)

    columnas_pacientes = []
    nombres_pacientes = []

    for carpeta in carpetas:
        ficheros = list(carpeta.glob("*gene_level_copy_number.v36.tsv"))
        if not ficheros:
            continue
        genes, valores = leer_fichero_paciente(ficheros[0])
        columnas_pacientes.append(valores)
        nombres_pacientes.append(carpeta.name)
        print("Leido " + carpeta.name + ": " + str(len(genes)) + " genes")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene_id"] + nombres_pacientes)
        for i in range(len(genes_referencia)):
            fila = [genes_referencia[i]]
            for columna in columnas_pacientes:
                fila.append(columna[i])
            w.writerow(fila)

    print("Guardado: " + str(out_path))
    print(str(len(genes_referencia)) + " genes x " + str(len(nombres_pacientes)) + " pacientes")
    return 0


if __name__ == "__main__":
    in_dir, out_path = sys.argv[1:3]
    sys.exit(main(in_dir, out_path))