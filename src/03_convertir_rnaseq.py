import csv, sys
from pathlib import Path


def leer_fichero_paciente(ruta):
    genes = []
    valores = []
    with ruta.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 9:
                continue
            gene_id = fila[0]
            gene_type = fila[2]
            tpm = fila[6]
            if gene_type == "protein_coding":
                genes.append(gene_id)
                valores.append(tpm)
    return genes, valores


def main(in_dir, out_path):
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    carpetas = [d for d in in_dir.iterdir() if d.is_dir()]

    # Usamos el primer paciente como referencia para la lista de genes
    primera_carpeta = carpetas[0]
    fichero_ref = list(primera_carpeta.glob("*augmented_star_gene_counts.tsv"))[0]
    genes_referencia, _ = leer_fichero_paciente(fichero_ref)

    columnas_pacientes = []
    nombres_pacientes = []

    for carpeta in carpetas:
        ficheros = list(carpeta.glob("*augmented_star_gene_counts.tsv"))
        if not ficheros:
            continue
        genes, valores = leer_fichero_paciente(ficheros[0])
        columnas_pacientes.append(valores)
        nombres_pacientes.append(carpeta.name)
        print(f"Leido {carpeta.name}: {len(genes)} genes")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["gene_id"] + nombres_pacientes)
        for i in range(len(genes_referencia)):
            fila = [genes_referencia[i]]
            for columna in columnas_pacientes:
                fila.append(columna[i])
            w.writerow(fila)

    print(f"Guardado: {out_path}")
    print(f"{len(genes_referencia)} genes x {len(nombres_pacientes)} pacientes")
    return 0


if __name__ == "__main__":
    in_dir, out_path = sys.argv[1:3]
    sys.exit(main(in_dir, out_path))