import sys, json, urllib.request
from pathlib import Path
import torch

# Bloque 8 (MIL, Seccion 4.5 del TFM): selecciona los N pacientes por
# cohorte para el volumen de MIL (alcance decidido en el paso 48: 12
# pacientes/cohorte, 60 en total, dentro de los 8 dias presupuestados).
#
# Universo de candidatos: la lista "pacientes" ya guardada en
# <COHORTE>_modelo_gat_gcn_final.pt (25_entrenar_gat_gcn_final.py), NO
# <COHORTE>_clinical_valid.tsv (esta ultima es la cohorte clinica
# completa de la Fase 1, sin cruzar con RNA-seq/CNV/metilacion ni con
# supervivencia utilizable para Cox - muchisimo mas grande, ver runlog
# de este paso). Se usa la lista del modelo final porque es
# exactamente el conjunto de pacientes con h_final ya calculado, el
# que hace falta para combinar z_WSI + embedding molecular (objetivo
# final del bloque: C-index de MIL+molecular vs. solo molecular).
#
# Metodo de verificacion de WSI, mismo rigor que la coordinacion de
# RNA-seq/CNV/metilacion de los Pasos 24-29: consulta a la API del GDC
# (no solo tamano de fichero), filtrando a muestra "Primary Tumor" y
# con desempate deterministico cuando hay mas de un fichero candidato
# por paciente (varias diapositivas escaneadas, o Diagnostic Slide +
# Tissue Slide simultaneos).

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def consultar_wsi_disponibles(case_ids):
    filtro = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }
    body = {
        "filters": filtro,
        "fields": "file_id,file_name,file_size,experimental_strategy,cases.submitter_id,cases.samples.sample_type",
        "size": "1000",
        "format": "JSON",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(GDC_FILES_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        resultado = json.loads(resp.read().decode("utf-8"))
    return resultado["data"]["hits"]


def elegir_mejor_wsi(candidatos_paciente):
    # candidatos_paciente: lista de hits (dict) de un unico case_id.
    # 1) solo Primary Tumor; 2) preferir Diagnostic Slide (DX) sobre
    # Tissue Slide (TS) - es el tipo ya usado en el paciente de prueba
    # TCGA-A1-A0SB (Pasos 44-46) y el estandar en literatura de MIL
    # sobre TCGA; 3) desempate final deterministico por file_id.
    primary = [
        h for h in candidatos_paciente
        if any(m.get("sample_type") == "Primary Tumor" for c in h.get("cases", []) for m in c.get("samples", []))
    ]
    if not primary:
        return None
    diagnostic = [h for h in primary if h.get("experimental_strategy") == "Diagnostic Slide"]
    pool = diagnostic if diagnostic else primary
    return min(pool, key=lambda h: h["file_id"])


def main(cohorte, ruta_modelo_gat_gcn, n_pacientes, ruta_salida):
    d = torch.load(ruta_modelo_gat_gcn, map_location="cpu", weights_only=False)
    candidatos = sorted(d["pacientes"])
    print(f"Cohorte: {cohorte}")
    print(f"  {len(candidatos)} candidatos con h_final ya calculado (modelo GAT+GCN final), orden alfabetico de case_id")

    hits = consultar_wsi_disponibles(candidatos)
    print(f"  {len(hits)} ficheros 'Slide Image' encontrados en el GDC para estos candidatos")

    por_paciente = {}
    for h in hits:
        for c in h.get("cases", []):
            sid = c.get("submitter_id")
            if sid in candidatos:
                por_paciente.setdefault(sid, []).append(h)

    seleccionados = []
    descartados_sin_wsi = []
    for case_id in candidatos:
        if len(seleccionados) >= n_pacientes:
            break
        mejor = elegir_mejor_wsi(por_paciente.get(case_id, []))
        if mejor is None:
            descartados_sin_wsi.append(case_id)
            continue
        seleccionados.append({
            "case_id": case_id,
            "file_id": mejor["file_id"],
            "file_name": mejor["file_name"],
            "file_size": mejor["file_size"],
            "experimental_strategy": mejor["experimental_strategy"],
        })

    print(f"  seleccionados: {len(seleccionados)}/{n_pacientes}")
    if descartados_sin_wsi:
        print(f"  descartados sin WSI Primary Tumor disponible ({len(descartados_sin_wsi)}): {descartados_sin_wsi}")

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("case_id\tfile_id\tfile_name\tfile_size\texperimental_strategy\n")
        for s in seleccionados:
            f.write(f"{s['case_id']}\t{s['file_id']}\t{s['file_name']}\t{s['file_size']}\t{s['experimental_strategy']}\n")
    print(f"Guardado: {ruta_salida}")

    if len(seleccionados) < n_pacientes:
        print(f"AVISO: solo se encontraron {len(seleccionados)} de {n_pacientes} pedidos, "
              f"se agotaron los {len(candidatos)} candidatos disponibles.")
        return 1
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_modelo_gat_gcn = argumentos[1]
    n_pacientes = int(argumentos[2]) if len(argumentos) > 2 else 12
    ruta_salida = argumentos[3] if len(argumentos) > 3 else f"data/processed/{cohorte}_pacientes_mil.tsv"
    sys.exit(main(cohorte, ruta_modelo_gat_gcn, n_pacientes, ruta_salida))
