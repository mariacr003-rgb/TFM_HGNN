import sys, json, urllib.request

# Verifica que las 3 tablas finales de una cohorte (rnaseq, cnv, metilacion)
# comparten exactamente los mismos pacientes (case_id), que los 3 comparten
# muestra tumoral primaria ("Primary Tumor") consistente, y que la metilacion
# es del array estandar del proyecto (Illumina 450K, ver Paso 11).
#
# No se fia de ningun fichero de seleccion previo: vuelve a consultar la API
# del GDC directamente sobre las cabeceras de las 3 tablas finales, porque un
# fichero de seleccion podria no reflejar lo que realmente se descargo. Asi
# paso desapercibido en BRCA (Paso 24) que 10 de los 200 pacientes tenian
# tipo de muestra inconsistente entre modalidades: la comprobacion de
# entonces solo verificaba que el case_id coincidiera entre tablas, no que
# el tipo de muestra fuera "Primary Tumor" en las 3 (ver Paso 29). La misma
# logica se aplica aqui al array de metilacion (Paso 31): no basta con haber
# filtrado por tamano de fichero al construir la lista de candidatos: se
# vuelve a comprobar el campo real "platform" sobre la tabla final.

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
TAMANO_LOTE = 500
ARRAY_METILACION_ESTANDAR = "Illumina Human Methylation 450"


def leer_cabecera(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        primera = f.readline().rstrip("\n")
    return primera.split("\t")[1:]


def consultar_lote(file_ids):
    filtro = {
        "op": "in",
        "content": {"field": "files.file_id", "value": file_ids},
    }
    body = {
        "filters": filtro,
        "fields": "file_id,cases.submitter_id,cases.samples.sample_type,platform",
        "size": str(len(file_ids)),
        "format": "JSON",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        GDC_FILES_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        resultado = json.loads(resp.read().decode("utf-8"))
    return resultado["data"]["hits"]


def resolver(file_ids):
    mapa = {}
    for i in range(0, len(file_ids), TAMANO_LOTE):
        lote = file_ids[i:i + TAMANO_LOTE]
        for hit in consultar_lote(lote):
            fid = hit["file_id"]
            casos = hit.get("cases", [])
            if not casos:
                continue
            case_id = casos[0]["submitter_id"]
            muestras = casos[0].get("samples", [])
            tipos = sorted(set(m["sample_type"] for m in muestras)) if muestras else []
            plataforma = hit.get("platform", "")
            mapa[fid] = (case_id, tipos, plataforma)
    return mapa


def main(ruta_rnaseq, ruta_cnv, ruta_meth):
    cab_rnaseq = leer_cabecera(ruta_rnaseq)
    cab_cnv = leer_cabecera(ruta_cnv)
    cab_meth = leer_cabecera(ruta_meth)
    print(f"Columnas de paciente: rnaseq={len(cab_rnaseq)}, cnv={len(cab_cnv)}, meth={len(cab_meth)}")

    mapa_rnaseq = resolver(cab_rnaseq)
    mapa_cnv = resolver(cab_cnv)
    mapa_meth = resolver(cab_meth)

    no_mapeados = [c for c in cab_rnaseq if c not in mapa_rnaseq]
    no_mapeados += [c for c in cab_cnv if c not in mapa_cnv]
    no_mapeados += [c for c in cab_meth if c not in mapa_meth]
    if no_mapeados:
        print(f"ATENCION: {len(no_mapeados)} columnas sin case_id resuelto por la API")
        return 1

    casos_rnaseq = {mapa_rnaseq[c][0] for c in cab_rnaseq}
    casos_cnv = {mapa_cnv[c][0] for c in cab_cnv}
    casos_meth = {mapa_meth[c][0] for c in cab_meth}

    mismo_case_id = casos_rnaseq == casos_cnv == casos_meth
    print(f"Los 3 tablas comparten exactamente el mismo conjunto de case_id: {mismo_case_id}")

    tipos_rnaseq = {mapa_rnaseq[c][0]: mapa_rnaseq[c][1] for c in cab_rnaseq}
    tipos_cnv = {mapa_cnv[c][0]: mapa_cnv[c][1] for c in cab_cnv}
    tipos_meth = {mapa_meth[c][0]: mapa_meth[c][1] for c in cab_meth}

    interseccion = casos_rnaseq & casos_cnv & casos_meth
    inconsistentes = []
    for case_id in sorted(interseccion):
        pr = "Primary Tumor" in tipos_rnaseq.get(case_id, [])
        pc = "Primary Tumor" in tipos_cnv.get(case_id, [])
        pm = "Primary Tumor" in tipos_meth.get(case_id, [])
        if not (pr and pc and pm):
            inconsistentes.append((case_id, tipos_rnaseq.get(case_id), tipos_cnv.get(case_id), tipos_meth.get(case_id)))

    print(f"Casos con 'Primary Tumor' consistente en las 3 modalidades: {len(interseccion) - len(inconsistentes)}/{len(interseccion)}")
    if inconsistentes:
        print(f"ATENCION: {len(inconsistentes)} casos SIN 'Primary Tumor' consistente:")
        for fila in inconsistentes:
            print(f"  {fila}")

    plataformas_meth = {mapa_meth[c][0]: mapa_meth[c][2] for c in cab_meth}
    array_distinto = [c for c in cab_meth if mapa_meth[c][2] != ARRAY_METILACION_ESTANDAR]
    array_correcto = not array_distinto
    print(f"Ficheros de metilacion con array '{ARRAY_METILACION_ESTANDAR}': {len(cab_meth) - len(array_distinto)}/{len(cab_meth)}")
    if array_distinto:
        print(f"ATENCION: {len(array_distinto)} ficheros de metilacion con array distinto del estandar:")
        for fid in array_distinto:
            print(f"  {fid}: {mapa_meth[fid][2]}")

    todo_ok = mismo_case_id and not inconsistentes and array_correcto
    print(f"\nVerificacion completa (case_id identicos, Primary Tumor consistente, array 450K): {todo_ok}")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    ruta_rnaseq, ruta_cnv, ruta_meth = sys.argv[1:4]
    sys.exit(main(ruta_rnaseq, ruta_cnv, ruta_meth))
