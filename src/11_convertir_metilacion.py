import csv, sys
from pathlib import Path
from contextlib import ExitStack
from itertools import zip_longest


def encontrar_fichero_metilacion(carpeta):
    ficheros = list(carpeta.glob("*methylation_array.sesame.level3betas.txt"))
    return ficheros[0] if ficheros else None


def main(in_dir, out_path):
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    carpetas = sorted(d for d in in_dir.iterdir() if d.is_dir())

    nombres_pacientes = []
    rutas_pacientes = []
    for carpeta in carpetas:
        ruta = encontrar_fichero_metilacion(carpeta)
        if ruta is None:
            continue
        nombres_pacientes.append(carpeta.name)
        rutas_pacientes.append(ruta)

    print(str(len(rutas_pacientes)) + " pacientes encontrados. Procesando en streaming (sin cargar todo en memoria)")

    n_sitios = 0
    with ExitStack() as pila, out_path.open("w", encoding="utf-8", newline="") as f_out:
        ficheros_abiertos = [pila.enter_context(ruta.open("r", encoding="utf-8")) for ruta in rutas_pacientes]

        w = csv.writer(f_out, delimiter="\t")
        w.writerow(["cg_id"] + nombres_pacientes)

        centinela = object()
        for lineas in zip_longest(*ficheros_abiertos, fillvalue=centinela):
            campos = []
            for i, linea in enumerate(lineas):
                if linea is centinela:
                    raise ValueError(
                        "El fichero de " + nombres_pacientes[i] +
                        " tiene menos sitios CpG que los demas (linea " + str(n_sitios + 1) + ")"
                    )
                campos.append(linea.rstrip("\n").split("\t"))

            cg_id_referencia = campos[0][0]
            fila = [cg_id_referencia]
            for i, campo in enumerate(campos):
                if campo[0] != cg_id_referencia:
                    raise ValueError(
                        "Desalineacion de sitios CpG: paciente " + nombres_pacientes[i] +
                        " tiene " + campo[0] + " en la posicion donde se esperaba " + cg_id_referencia
                    )
                fila.append(campo[1])

            w.writerow(fila)
            n_sitios += 1
            if n_sitios % 50000 == 0:
                print(str(n_sitios) + " sitios CpG procesados...")

    print("Guardado: " + str(out_path))
    print(str(n_sitios) + " sitios CpG x " + str(len(nombres_pacientes)) + " pacientes")
    return 0


if __name__ == "__main__":
    in_dir, out_path = sys.argv[1:3]
    sys.exit(main(in_dir, out_path))