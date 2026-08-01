import csv, sys

# A partir de una lista de pacientes ya coordinados (case_id + file_id por
# modalidad, salida de 15_interseccion_pacientes.py), genera los 3 manifests
# filtrados en formato GDC (id, filename, md5, size, state) que
# tools/gdc-client.exe necesita para descargar exactamente esos ficheros.


def leer_manifest_completo(ruta):
    filas_por_id = {}
    with open(ruta, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for fila in reader:
            filas_por_id[fila[0]] = fila
    return header, filas_por_id


def leer_seleccion(ruta):
    seleccion = []
    with open(ruta, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for fila in reader:
            seleccion.append(fila)
    return seleccion


def escribir_manifest(ruta_salida, header, filas):
    with open(ruta_salida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(filas)


def main(ruta_manifest_full, ruta_seleccion, out_rnaseq, out_cnv, out_meth):
    header, filas_por_id = leer_manifest_completo(ruta_manifest_full)
    seleccion = leer_seleccion(ruta_seleccion)

    filas_rnaseq = [filas_por_id[fid_rnaseq] for _, fid_rnaseq, _, _ in seleccion]
    filas_cnv = [filas_por_id[fid_cnv] for _, _, fid_cnv, _ in seleccion]
    filas_meth = [filas_por_id[fid_meth] for _, _, _, fid_meth in seleccion]

    escribir_manifest(out_rnaseq, header, filas_rnaseq)
    escribir_manifest(out_cnv, header, filas_cnv)
    escribir_manifest(out_meth, header, filas_meth)

    print(f"{out_rnaseq}: {len(filas_rnaseq)} ficheros")
    print(f"{out_cnv}: {len(filas_cnv)} ficheros")
    print(f"{out_meth}: {len(filas_meth)} ficheros")
    return 0


if __name__ == "__main__":
    ruta_manifest_full, ruta_seleccion = sys.argv[1], sys.argv[2]
    out_rnaseq, out_cnv, out_meth = sys.argv[3], sys.argv[4], sys.argv[5]
    sys.exit(main(ruta_manifest_full, ruta_seleccion, out_rnaseq, out_cnv, out_meth))
