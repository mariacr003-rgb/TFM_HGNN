import csv, sys, json, urllib.request
from pathlib import Path
import torch

# Construye, para una cohorte ya coordinada (200 pacientes, ver Pasos 24-31),
# un tensor de atributos por paciente para los nodos "gen" del grafo
# heterogeneo (data/processed/grafo_heterogeneo.pt): 3 canales por gen
# (expresion RNA-seq, numero de copias CNV, metilacion del promotor).
#
# El grafo NO se toca en este script (su topologia es compartida entre
# pacientes): la salida es un fichero aparte, pequeno, con solo los
# atributos, para combinarlo con el grafo compartido en el script de
# entrenamiento (reutilizando el mismo edge_index para todos los
# pacientes, sin duplicar la topologia 200 veces).
#
# Problema principal: RNA-seq, CNV y metilacion NO estan alineados entre
# si de forma trivial:
#   - RNA-seq: 19.962 filas, SOLO genes protein_coding, en el mismo orden
#     exacto que los 19.962 nodos "gen" del grafo (verificado: mismo
#     orden en las 5 cohortes). Se puede usar por POSICION de fila.
#   - CNV: 60.624 filas, TODOS los genes del pipeline ASCAT2 (no filtra
#     por tipo). Su orden NO coincide con el de RNA-seq/grafo: hay que
#     unir por ENSG (sin version), no por posicion. 13 genes
#     protein_coding (todos mitocondriales: MT-ND1, MT-CO1, etc.) no
#     tienen fila en CNV -> quedan sin dato (ASCAT2 no cubre ADN
#     mitocondrial). 44 ENSG aparecen duplicados en CNV (genes de la
#     region pseudoautosomal, PAR); nos quedamos con la primera fila
#     vista de cada uno (mismo criterio "primeros N" ya usado en el
#     resto del proyecto).
#   - Metilacion: 486.427 sitios CpG, indexados por sonda (cg_id), no por
#     gen. Hace falta un fichero externo de anotacion sonda->gen
#     (RUTA_MANIFEST_METILACION, ver docs/manifest-datos.tsv) para saber
#     que gen(es) esta cerca de cada sonda, y agregar (media) las sondas
#     de la region del promotor (|distancia a inicio de transcripcion
#     protein_coding| <= 1500, definicion estandar TSS1500) en un unico
#     valor por gen y paciente.
#
# Los 18 genes PAR duplicados en la lista de nodos (mismo ENSG, una fila
# "normal" y otra con sufijo _PAR_Y, ver Paso 23) NO se tratan con codigo
# especial: como CNV y metilacion se unen por ENSG (no por posicion),
# las 2 posiciones de nodo de un mismo ENSG buscan la MISMA clave en los
# diccionarios de CNV/metilacion y reciben automaticamente el MISMO
# valor. Es una limitacion heredada de que ni CNV ni la anotacion de
# metilacion distinguen la copia X de la copia Y del gen, no un error de
# este script. RNA-seq si distingue las 2 copias (union posicional).
#
# Valores no disponibles (los 13 genes mitocondriales en CNV, genes sin
# ninguna sonda de promotor en el array) se guardan como NaN, con un
# tensor de mascara de validez aparte; la imputacion, si hace falta, se
# decide en el script de entrenamiento, no aqui.

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
TAMANO_LOTE = 500

RUTA_MANIFEST_METILACION = Path("data/raw/HM450_manifest_gencode_v41.tsv.gz")
DISTANCIA_PROMOTOR_MAX = 1500  # TSS1500, definicion estandar de region promotora


def resolver_case_id(file_ids):
    # Igual patron que 14_mapear_case_id.py / 17_verificar_coordinacion.py,
    # pero solo pide case_id (no sample_type/platform: la coordinacion ya
    # se verifico por separado con 17_verificar_coordinacion.py).
    mapa = {}
    for i in range(0, len(file_ids), TAMANO_LOTE):
        lote = file_ids[i:i + TAMANO_LOTE]
        filtro = {"op": "in", "content": {"field": "files.file_id", "value": lote}}
        body = {
            "filters": filtro,
            "fields": "file_id,cases.submitter_id",
            "size": str(len(lote)),
            "format": "JSON",
        }
        req = urllib.request.Request(
            GDC_FILES_URL, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resultado = json.loads(resp.read().decode("utf-8"))
        for hit in resultado["data"]["hits"]:
            fid = hit["file_id"]
            casos = hit.get("cases", [])
            if casos:
                mapa[fid] = casos[0]["submitter_id"]
    return mapa


def leer_cabecera(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        primera = f.readline().rstrip("\n")
    return primera.split("\t")[1:]


def calcular_orden_columnas(cabecera, mapa_fid_a_caso, pacientes):
    # Para cada paciente (en el orden final ya decidido), en que columna
    # (indice) del fichero original esta su dato.
    columna_por_caso = {mapa_fid_a_caso[fid]: i for i, fid in enumerate(cabecera)}
    return [columna_por_caso[caso] for caso in pacientes]


def convertir_a_float(valor_str):
    # Algunas filas de CNV tienen copy_number vacio en el fichero crudo
    # del GDC (genes en regiones no cubiertas por la segmentacion de
    # ASCAT2, ~0,6-0,7% de los genes, ver Paso 32): no es un error, se
    # trata como dato no disponible (NaN), no como fallo de conversion.
    try:
        return float(valor_str)
    except ValueError:
        return float("nan")


def cargar_tabla_numerica(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabecera
        ids_fila = []
        matriz = []
        for fila in reader:
            ids_fila.append(fila[0])
            matriz.append([convertir_a_float(v) for v in fila[1:]])
    return ids_fila, matriz


def cargar_universo_genes(ruta_rnaseq):
    # Mismo criterio que 13_construir_grafo.py: genes en el orden de fila
    # de RNA-seq (ya filtrado a protein_coding al generar la tabla), sin
    # version de ENSG. Puede tener ENSG repetidos (18 genes PAR).
    ids_fila, _matriz = cargar_tabla_numerica(ruta_rnaseq)
    genes_orden = [g.split(".")[0] for g in ids_fila]
    return genes_orden


def construir_mapa_simbolo_ensg(ruta_rnaseq_crudo):
    # Reconstruye ensg_a_simbolo desde un fichero crudo STAR de referencia
    # (igual que 13_construir_grafo.py), y lo invierte a simbolo->ENSG
    # excluyendo simbolos ambiguos (compartidos por mas de un ENSG, ej.
    # genes PAR con simbolo repetido entre su copia X e Y).
    ensg_a_simbolo = {}
    with ruta_rnaseq_crudo.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 9:
                continue
            gene_id, gene_name, gene_type = fila[0], fila[1], fila[2]
            if gene_type != "protein_coding":
                continue
            ensg_a_simbolo[gene_id.split(".")[0]] = gene_name

    simbolo_a_ensgs = {}
    for ensg, simbolo in ensg_a_simbolo.items():
        simbolo_a_ensgs.setdefault(simbolo, []).append(ensg)

    simbolo_a_ensg = {s: l[0] for s, l in simbolo_a_ensgs.items() if len(l) == 1}
    contadores = {
        "simbolos_ambiguos_excluidos": sum(1 for l in simbolo_a_ensgs.values() if len(l) > 1),
    }
    return simbolo_a_ensg, contadores


def encontrar_fichero_rnaseq_crudo(dir_rnaseq_cohorte):
    carpetas = [d for d in dir_rnaseq_cohorte.iterdir() if d.is_dir()]
    return list(carpetas[0].glob("*augmented_star_gene_counts.tsv"))[0]


def construir_canal_rnaseq(ruta_rnaseq, genes_orden, cabecera, orden_columnas):
    ids_fila, matriz = cargar_tabla_numerica(ruta_rnaseq)
    ids_fila_sin_version = [g.split(".")[0] for g in ids_fila]
    if ids_fila_sin_version != genes_orden:
        raise ValueError(
            "El orden de genes de RNA-seq no coincide con el universo de genes "
            "del grafo; no se puede alinear por posicion de fila."
        )
    n_genes = len(genes_orden)
    n_pacientes = len(orden_columnas)
    canal = [[matriz[i][orden_columnas[j]] for j in range(n_pacientes)] for i in range(n_genes)]
    valido = [[canal[i][j] == canal[i][j] for j in range(n_pacientes)] for i in range(n_genes)]  # x==x es False solo para NaN
    return canal, valido


def construir_ensg_a_valores_cnv(ruta_cnv):
    ids_fila, matriz = cargar_tabla_numerica(ruta_cnv)
    ensg_a_valores = {}
    duplicados = 0
    for id_fila, valores in zip(ids_fila, matriz):
        ensg = id_fila.split(".")[0]
        if ensg in ensg_a_valores:
            duplicados += 1
            continue
        ensg_a_valores[ensg] = valores
    contadores = {
        "filas_totales": len(ids_fila),
        "ensg_unicos": len(ensg_a_valores),
        "filas_duplicadas_descartadas": duplicados,
    }
    return ensg_a_valores, contadores


def construir_canal_desde_ensg(genes_orden, ensg_a_valores, orden_columnas):
    n_genes = len(genes_orden)
    n_pacientes = len(orden_columnas)
    canal = [[float("nan")] * n_pacientes for _ in range(n_genes)]
    valido = [[False] * n_pacientes for _ in range(n_genes)]
    genes_sin_dato = 0
    for i, ensg in enumerate(genes_orden):
        valores_originales = ensg_a_valores.get(ensg)
        if valores_originales is None:
            genes_sin_dato += 1
            continue
        for j in range(n_pacientes):
            v = valores_originales[orden_columnas[j]]
            canal[i][j] = v
            valido[i][j] = (v == v)  # x==x es False solo para NaN
    return canal, valido, genes_sin_dato


def construir_mapa_sonda_a_ensg(ruta_manifest, simbolo_a_ensg, genes_validos):
    # Formato del manifest (sesame/Zhou Lab, HM450 + GENCODE v41):
    # CpG_chrm CpG_beg CpG_end probe_strand probeID genesUniq geneNames
    # transcriptTypes transcriptIDs distToTSS
    # geneNames/transcriptTypes/distToTSS: 1 entrada por transcrito
    # solapante, separadas por ";" (mismo numero de entradas en las 3).
    import gzip

    mapa = {}
    contadores = {
        "sin_gen_asignado": 0,
        "simbolo_no_mapeado_a_ensg": 0,
        "sin_sonda_en_promotor_protein_coding": 0,
        "asignadas_a_promotor": 0,
    }
    with gzip.open(ruta_manifest, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabecera
        for fila in reader:
            if len(fila) < 10:
                continue
            cg_id = fila[4]
            gen_unico = fila[5]
            if gen_unico == "NA":
                contadores["sin_gen_asignado"] += 1
                continue
            ensg = simbolo_a_ensg.get(gen_unico)
            if ensg is None or ensg not in genes_validos:
                contadores["simbolo_no_mapeado_a_ensg"] += 1
                continue

            gene_names = fila[6].split(";")
            transcript_types = fila[7].split(";")
            dist_tss = fila[9].split(";")
            es_promotor = False
            for gn, tt, dist in zip(gene_names, transcript_types, dist_tss):
                if gn != gen_unico or tt != "protein_coding":
                    continue
                try:
                    d = int(dist)
                except ValueError:
                    continue
                if abs(d) <= DISTANCIA_PROMOTOR_MAX:
                    es_promotor = True
                    break
            if not es_promotor:
                contadores["sin_sonda_en_promotor_protein_coding"] += 1
                continue

            mapa[cg_id] = ensg
            contadores["asignadas_a_promotor"] += 1

    return mapa, contadores


def acumular_metilacion(ruta_meth, mapa_sonda_a_ensg, n_pacientes_original):
    sumas = {}
    contadores_validos = {}
    n_procesadas = 0
    with open(ruta_meth, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabecera
        for fila in reader:
            cg_id = fila[0]
            n_procesadas += 1
            ensg = mapa_sonda_a_ensg.get(cg_id)
            if ensg is not None:
                if ensg not in sumas:
                    sumas[ensg] = [0.0] * n_pacientes_original
                    contadores_validos[ensg] = [0] * n_pacientes_original
                for j, valor_str in enumerate(fila[1:]):
                    try:
                        valor = float(valor_str)
                    except ValueError:
                        continue
                    sumas[ensg][j] += valor
                    contadores_validos[ensg][j] += 1
            if n_procesadas % 50000 == 0:
                print(f"  {n_procesadas} sitios CpG procesados...")

    ensg_a_medias = {}
    for ensg, suma_lista in sumas.items():
        ensg_a_medias[ensg] = [
            (suma_lista[j] / contadores_validos[ensg][j]) if contadores_validos[ensg][j] > 0 else None
            for j in range(n_pacientes_original)
        ]
    return ensg_a_medias


def construir_canal_metilacion(genes_orden, ensg_a_medias, orden_columnas):
    n_genes = len(genes_orden)
    n_pacientes = len(orden_columnas)
    canal = [[float("nan")] * n_pacientes for _ in range(n_genes)]
    valido = [[False] * n_pacientes for _ in range(n_genes)]
    genes_sin_dato = 0
    for i, ensg in enumerate(genes_orden):
        medias_originales = ensg_a_medias.get(ensg)
        if medias_originales is None:
            genes_sin_dato += 1
            continue
        alguna_valida = False
        for j in range(n_pacientes):
            media = medias_originales[orden_columnas[j]]
            if media is not None:
                canal[i][j] = media
                valido[i][j] = True
                alguna_valida = True
        if not alguna_valida:
            genes_sin_dato += 1
    return canal, valido, genes_sin_dato


def main(cohorte):
    dir_datos = Path("data/raw")
    ruta_rnaseq = dir_datos / f"{cohorte}_rnaseq.tsv"
    ruta_cnv = dir_datos / f"{cohorte}_cnv.tsv"
    ruta_meth = dir_datos / f"{cohorte}_metilacion.tsv"
    dir_rnaseq_crudo = dir_datos / f"rnaseq_{cohorte.lower()}"
    ruta_salida = Path("data/processed") / f"{cohorte}_atributos_gen.pt"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cohorte: {cohorte}")
    print("Cargando universo de genes (orden del grafo, RNA-seq protein_coding)...")
    genes_orden = cargar_universo_genes(ruta_rnaseq)
    print(f"  {len(genes_orden)} genes ({len(set(genes_orden))} ENSG unicos)")

    print("Resolviendo columnas de paciente (case_id) via API del GDC...")
    cab_rnaseq = leer_cabecera(ruta_rnaseq)
    cab_cnv = leer_cabecera(ruta_cnv)
    cab_meth = leer_cabecera(ruta_meth)
    mapa_rnaseq = resolver_case_id(cab_rnaseq)
    mapa_cnv = resolver_case_id(cab_cnv)
    mapa_meth = resolver_case_id(cab_meth)

    casos_rnaseq = {mapa_rnaseq[c] for c in cab_rnaseq}
    casos_cnv = {mapa_cnv[c] for c in cab_cnv}
    casos_meth = {mapa_meth[c] for c in cab_meth}
    if not (casos_rnaseq == casos_cnv == casos_meth):
        raise ValueError(
            "Las 3 tablas no comparten el mismo conjunto de pacientes. "
            "Verificar coordinacion con 17_verificar_coordinacion.py antes de continuar."
        )
    pacientes = sorted(casos_rnaseq)
    print(f"  {len(pacientes)} pacientes coordinados en las 3 modalidades")

    orden_cols_rnaseq = calcular_orden_columnas(cab_rnaseq, mapa_rnaseq, pacientes)
    orden_cols_cnv = calcular_orden_columnas(cab_cnv, mapa_cnv, pacientes)
    orden_cols_meth = calcular_orden_columnas(cab_meth, mapa_meth, pacientes)

    print("Canal 1/3: RNA-seq (alineado por posicion de fila)...")
    canal_rnaseq, valido_rnaseq = construir_canal_rnaseq(ruta_rnaseq, genes_orden, cab_rnaseq, orden_cols_rnaseq)

    print("Canal 2/3: CNV (alineado por ENSG)...")
    ensg_a_valores_cnv, contadores_cnv = construir_ensg_a_valores_cnv(ruta_cnv)
    print(f"  {contadores_cnv}")
    canal_cnv, valido_cnv, genes_sin_cnv = construir_canal_desde_ensg(genes_orden, ensg_a_valores_cnv, orden_cols_cnv)
    print(f"  genes del universo sin dato de CNV: {genes_sin_cnv}")

    print("Canal 3/3: metilacion (agregada por promotor, TSS1500)...")
    print("  Cargando mapa sonda->gen desde el manifest de metilacion...")
    simbolo_a_ensg, contadores_simbolo = construir_mapa_simbolo_ensg(encontrar_fichero_rnaseq_crudo(dir_rnaseq_crudo))
    print(f"  {contadores_simbolo}")
    mapa_sonda_a_ensg, contadores_manifest = construir_mapa_sonda_a_ensg(
        RUTA_MANIFEST_METILACION, simbolo_a_ensg, set(genes_orden)
    )
    print(f"  {contadores_manifest}")
    print("  Procesando metilacion en streaming (486427 sitios CpG, sin cargar todo en memoria)...")
    ensg_a_medias_meth = acumular_metilacion(ruta_meth, mapa_sonda_a_ensg, len(cab_meth))
    canal_meth, valido_meth, genes_sin_meth = construir_canal_metilacion(genes_orden, ensg_a_medias_meth, orden_cols_meth)
    print(f"  genes del universo sin ninguna sonda de promotor: {genes_sin_meth}")

    print("Construyendo tensor final [pacientes, genes, canales]...")
    # Conversion vectorizada (torch.tensor sobre listas anidadas) en vez de
    # asignar elemento a elemento (evita ~12 millones de asignaciones
    # individuales, muy lentas en CPU).
    canales_x = torch.stack([
        torch.tensor(canal_rnaseq, dtype=torch.float32),
        torch.tensor(canal_cnv, dtype=torch.float32),
        torch.tensor(canal_meth, dtype=torch.float32),
    ], dim=-1)  # [genes, pacientes, 3]
    canales_valido = torch.stack([
        torch.tensor(valido_rnaseq, dtype=torch.bool),
        torch.tensor(valido_cnv, dtype=torch.bool),
        torch.tensor(valido_meth, dtype=torch.bool),
    ], dim=-1)  # [genes, pacientes, 3]
    x = canales_x.permute(1, 0, 2).contiguous()  # [pacientes, genes, 3]
    mascara_valido = canales_valido.permute(1, 0, 2).contiguous()

    salida = {
        "cohorte": cohorte,
        "pacientes": pacientes,
        "genes_ensg": genes_orden,
        "canales": ["rnaseq_tpm", "cnv_copy_number", "metilacion_promotor_beta"],
        "x": x,
        "mascara_valido": mascara_valido,
    }
    torch.save(salida, ruta_salida)

    print()
    print(f"Guardado: {ruta_salida}")
    print(f"Tensor x: {tuple(x.shape)}, validos: {mascara_valido.sum().item()} / {mascara_valido.numel()}")
    return 0


if __name__ == "__main__":
    cohorte = sys.argv[1]
    sys.exit(main(cohorte))
