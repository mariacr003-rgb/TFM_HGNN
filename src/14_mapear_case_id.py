import sys, json, urllib.request

# El manifest completo del GDC (columnas id, filename, md5, size, state) no
# incluye el case_id (paciente) de cada fichero: en RNA-seq y metilacion el
# nombre de fichero es un UUID sin relacion visible con el paciente, y en
# CNV el UUID incrustado en el nombre tampoco es el case_id. Por eso hace
# falta resolverlo consultando la API REST del GDC (endpoint /files), por
# lotes de file_id, pidiendo tambien el tipo de muestra (sample_type) para
# poder distinguir despues tumor primario de tejido normal/metastasico.

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
TAMANO_LOTE = 500


def leer_lista(ruta):
    # Formato esperado: 2 columnas separadas por tab, sin cabecera:
    # file_id (columna "id" del manifest completo) y filename.
    # Se puede generar con, por ejemplo:
    #   awk -F'\t' '$2 ~ /<patron>/ {print $1"\t"$2}' manifest_full.txt > lista.tsv
    filas = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 2:
                continue
            filas.append((partes[0], partes[1]))
    return filas


def consultar_lote(file_ids):
    filtro = {
        "op": "in",
        "content": {"field": "files.file_id", "value": file_ids},
    }
    body = {
        "filters": filtro,
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
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


def mapear(ruta_lista):
    filas = leer_lista(ruta_lista)
    mapa = {}
    for i in range(0, len(filas), TAMANO_LOTE):
        lote = filas[i:i + TAMANO_LOTE]
        file_ids = [fid for fid, _ in lote]
        hits = consultar_lote(file_ids)
        for hit in hits:
            fid = hit["file_id"]
            casos = hit.get("cases", [])
            if not casos:
                continue
            case_submitter_id = casos[0]["submitter_id"]
            muestras = casos[0].get("samples", [])
            tipos_muestra = sorted(set(m["sample_type"] for m in muestras)) if muestras else []
            mapa[fid] = (case_submitter_id, ";".join(tipos_muestra))
        print(f"  lote {i}-{i + len(lote)}: {len(hits)} ficheros resueltos", file=sys.stderr)
    return filas, mapa


def main(ruta_lista, ruta_salida):
    filas, mapa = mapear(ruta_lista)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        for fid, filename in filas:
            case_id, tipo_muestra = mapa.get(fid, ("", ""))
            f.write(f"{fid}\t{filename}\t{case_id}\t{tipo_muestra}\n")
    resueltos = sum(1 for v in mapa.values() if v[0])
    print(f"Guardado: {ruta_salida} ({len(filas)} ficheros, {resueltos} con case_id resuelto)")
    return 0


if __name__ == "__main__":
    ruta_lista, ruta_salida = sys.argv[1], sys.argv[2]
    sys.exit(main(ruta_lista, ruta_salida))
