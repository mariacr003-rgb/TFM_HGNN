import csv, sys
from pathlib import Path


def leer_fichero_paciente(ruta):
    sitios = []
    valores = []
    with ruta.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 2:
                continue
            cg_id = fila[0]
            valor_beta = fila[1]
            sitios.append(cg_id)
            valores.append(valor_beta)
    return sitios, valores


def main(in_dir, out_path):
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    carpetas = [d for d in in_dir.iterdir() if d.is_dir()]

    primera_carpeta = carpetas[0]
    fichero_ref = list(primera_carpeta.glob("*methylation_array.sesame.level3betas.txt"))[0]
    sitios_referencia, _ = leer_fichero_paciente(fichero_ref)

    columnas_pacientes = []
    nombres_pacientes = []

    for carpeta in carpetas:
        ficheros = list(carpeta.glob("*methylation_array.sesame.level3betas.txt"))
        if not ficheros:
            continue
        sitios, valores = leer_fichero_paciente(ficheros[0])
        columnas_pacientes.append(valores)
        nombres_pacientes.append(carpeta.name)
        print("Leido " + carpeta.name + ": " + str(len(sitios)) + " sitios CpG")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cg_id"] + nombres_pacientes)
        for i in range(len(sitios_referencia)):
            fila = [sitios_referencia[i]]
            for columna in columnas_pacientes:
                fila.append(columna[i])
            w.writerow(fila)

    print("Guardado: " + str(out_path))
    print(str(len(sitios_referencia)) + " sitios CpG x " + str(len(nombres_pacientes)) + " pacientes")
    return 0


if __name__ == "__main__":
    in_dir, out_path = sys.argv[1:3]
    sys.exit(main(in_dir, out_path))