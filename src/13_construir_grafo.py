import csv, sys, gzip
from pathlib import Path
import torch
from torch_geometric.data import HeteroData

# Fichero de RNA-seq de referencia para definir el universo de nodos Gen
# (mismo conjunto de 19962 genes protein_coding en las 5 cohortes, mismo
# pipeline GENCODE v36; se usa BRCA por ser la primera cohorte descargada)
DIR_RNASEQ_REF = Path("data/raw/rnaseq_brca")

RUTA_BIOMART = Path("data/raw/gen_proteina_conversion.txt")
RUTA_STRING = Path("data/raw/9606.protein.links.v12.0.txt.gz")
RUTA_REACTOME_PATHWAYS = Path("data/raw/ReactomePathways.txt")
RUTA_REACTOME_GEN_VIA = Path("data/raw/Ensembl2Reactome_All_Levels.txt")
RUTA_KEGG_PATHWAYS = Path("data/raw/kegg_pathways_hsa.txt")
RUTA_KEGG_GEN_VIA = Path("data/raw/kegg_gene_pathway_hsa.txt")
RUTA_KEGG_GENES = Path("data/raw/kegg_genes_hsa.txt")
RUTA_TRRUST = Path("data/raw/trrust_rawdata.human.tsv")

RUTA_SALIDA = Path("data/processed/grafo_heterogeneo.pt")

UMBRAL_STRING = 700

MODO_REGULACION_A_CODIGO = {"Unknown": 0, "Activation": 1, "Repression": 2}


def cargar_universo_genes():
    carpetas = [d for d in DIR_RNASEQ_REF.iterdir() if d.is_dir()]
    fichero_ref = list(carpetas[0].glob("*augmented_star_gene_counts.tsv"))[0]

    ensg_a_simbolo = {}
    genes_orden = []
    with fichero_ref.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 9:
                continue
            gene_id, gene_name, gene_type = fila[0], fila[1], fila[2]
            if gene_type != "protein_coding":
                continue
            ensg = gene_id.split(".")[0]
            ensg_a_simbolo[ensg] = gene_name
            genes_orden.append(ensg)

    return genes_orden, ensg_a_simbolo


def construir_mapa_simbolo_ensg(ensg_a_simbolo):
    # Diccionario inverso simbolo -> ENSG, excluyendo los simbolos
    # ambiguos (compartidos por mas de un ENSG, ej. genes de la region
    # pseudoautosomal X/Y como ASMT/ASMTL/CD99), para no asignar
    # arbitrariamente las aristas de TRRUST/KEGG a uno de los dos genes.
    simbolo_a_ensgs = {}
    for ensg, simbolo in ensg_a_simbolo.items():
        simbolo_a_ensgs.setdefault(simbolo, []).append(ensg)

    simbolo_a_ensg = {}
    simbolos_ambiguos = []
    for simbolo, lista_ensg in simbolo_a_ensgs.items():
        if len(lista_ensg) == 1:
            simbolo_a_ensg[simbolo] = lista_ensg[0]
        else:
            simbolos_ambiguos.append((simbolo, lista_ensg))

    contadores = {
        "simbolos_ambiguos_excluidos": len(simbolos_ambiguos),
        "genes_afectados_por_ambiguedad": sum(len(l) for _s, l in simbolos_ambiguos),
    }
    return simbolo_a_ensg, simbolos_ambiguos, contadores


def cargar_mapa_ensg_ensp(genes_indice):
    # BioMart: Gene stable ID -> Protein stable ID. Puede haber varias
    # proteinas (isoformas) por gen; se conservan todas las que tengan
    # Protein stable ID no vacio.
    ensg_a_ensp = {}
    ensp_orden = []
    ensp_vistos = set()
    contadores = {"ensp_vacio": 0, "ensg_fuera_del_universo": 0}
    with RUTA_BIOMART.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabecera
        for fila in reader:
            if len(fila) < 2:
                continue
            ensg, ensp = fila[0], fila[1]
            if not ensp:
                contadores["ensp_vacio"] += 1
                continue
            if ensg not in genes_indice:
                contadores["ensg_fuera_del_universo"] += 1
                continue
            ensg_a_ensp.setdefault(ensg, []).append(ensp)
            if ensp not in ensp_vistos:
                ensp_vistos.add(ensp)
                ensp_orden.append(ensp)

    return ensg_a_ensp, ensp_orden, contadores


def cargar_aristas_ppi(ensp_indice, umbral):
    # STRING: "protein1 protein2 combined_score", separado por espacio,
    # con prefijo de especie "9606." que hay que quitar para casar con
    # los IDs de BioMart.
    aristas = []
    contadores = {"score_bajo_umbral": 0, "proteina_no_mapeada": 0}
    with gzip.open(RUTA_STRING, "rt", encoding="utf-8") as f:
        next(f)  # cabecera
        for linea in f:
            p1, p2, score = linea.split()
            score = int(score)
            if score < umbral:
                contadores["score_bajo_umbral"] += 1
                continue
            p1 = p1.replace("9606.", "")
            p2 = p2.replace("9606.", "")
            if p1 not in ensp_indice or p2 not in ensp_indice:
                contadores["proteina_no_mapeada"] += 1
                continue
            aristas.append((ensp_indice[p1], ensp_indice[p2], score))

    return aristas, contadores


def cargar_catalogo_reactome():
    # Catalogo completo de vias humanas de Reactome (nodos), independiente
    # de si acaban teniendo genes asociados en nuestro subconjunto.
    vias_orden = []
    with RUTA_REACTOME_PATHWAYS.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 3:
                continue
            via_id, _nombre, especie = fila
            if especie != "Homo sapiens":
                continue
            vias_orden.append(via_id)

    return vias_orden


def cargar_aristas_reactome(genes_indice, vias_indice):
    # Solo se usan las filas cuyo id de origen ya viene en ENSG directo
    # (no ENST/ENSP), segun lo acordado.
    aristas = []
    contadores = {
        "especie_no_humana": 0,
        "id_origen_no_ensg": 0,
        "ensg_fuera_del_universo": 0,
        "via_fuera_del_catalogo": 0,
    }
    with RUTA_REACTOME_GEN_VIA.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 6:
                continue
            origen_id, via_id = fila[0], fila[1]
            especie = fila[5]
            if especie != "Homo sapiens":
                contadores["especie_no_humana"] += 1
                continue
            if not origen_id.startswith("ENSG"):
                contadores["id_origen_no_ensg"] += 1
                continue
            ensg = origen_id.split(".")[0]
            if ensg not in genes_indice:
                contadores["ensg_fuera_del_universo"] += 1
                continue
            if via_id not in vias_indice:
                contadores["via_fuera_del_catalogo"] += 1
                continue
            aristas.append((ensg, via_id))

    return aristas, contadores


def cargar_catalogo_kegg():
    vias_orden = []
    with RUTA_KEGG_PATHWAYS.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 2:
                continue
            vias_orden.append(fila[0])

    return vias_orden


def cargar_mapa_entrez_simbolo():
    # kegg_genes_hsa.txt: Entrez ID, tipo, posicion, descripcion. Se
    # queda solo con tipo "CDS" y extrae el simbolo como el primer token
    # antes de la primera coma, dentro de la parte anterior al primer
    # punto y coma; los genes sin simbolo oficial no tienen punto y coma
    # en la descripcion y se descartan.
    entrez_a_simbolo = {}
    contadores = {"no_es_cds": 0, "sin_simbolo_en_descripcion": 0}
    with RUTA_KEGG_GENES.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 4:
                continue
            entrez_id, tipo, _posicion, descripcion = fila
            if tipo != "CDS":
                contadores["no_es_cds"] += 1
                continue
            if ";" not in descripcion:
                contadores["sin_simbolo_en_descripcion"] += 1
                continue
            parte_simbolos = descripcion.split(";")[0]
            simbolo = parte_simbolos.split(",")[0].strip()
            entrez_a_simbolo[entrez_id] = simbolo

    return entrez_a_simbolo, contadores


def cargar_aristas_kegg(entrez_a_simbolo, simbolo_a_ensg, genes_indice, vias_indice):
    aristas = []
    contadores = {
        "via_fuera_del_catalogo": 0,
        "entrez_sin_simbolo": 0,
        "simbolo_no_mapeado_a_ensg": 0,
    }
    with RUTA_KEGG_GEN_VIA.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 2:
                continue
            entrez_id, via_id_completo = fila
            via_id = via_id_completo.replace("path:", "")
            if via_id not in vias_indice:
                contadores["via_fuera_del_catalogo"] += 1
                continue
            simbolo = entrez_a_simbolo.get(entrez_id)
            if simbolo is None:
                contadores["entrez_sin_simbolo"] += 1
                continue
            ensg = simbolo_a_ensg.get(simbolo)
            if ensg is None or ensg not in genes_indice:
                contadores["simbolo_no_mapeado_a_ensg"] += 1
                continue
            aristas.append((ensg, via_id))

    return aristas, contadores


def cargar_aristas_regulacion(simbolo_a_ensg, genes_indice):
    # TRRUST: factor de transcripcion, gen diana, modo de regulacion, PMID.
    # Arista dirigida TF -> diana.
    aristas = []
    contadores = {"tf_simbolo_no_mapeado": 0, "diana_simbolo_no_mapeado": 0}
    with RUTA_TRRUST.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for fila in reader:
            if len(fila) < 4:
                continue
            tf, diana, modo, _pmid = fila
            ensg_tf = simbolo_a_ensg.get(tf)
            ensg_diana = simbolo_a_ensg.get(diana)
            if ensg_tf is None or ensg_tf not in genes_indice:
                contadores["tf_simbolo_no_mapeado"] += 1
                continue
            if ensg_diana is None or ensg_diana not in genes_indice:
                contadores["diana_simbolo_no_mapeado"] += 1
                continue
            aristas.append((ensg_tf, ensg_diana, modo))

    return aristas, contadores


def imprimir_contadores(titulo, contadores):
    print(f"  Descartes en {titulo}:")
    for motivo, n in contadores.items():
        print(f"    {motivo}: {n}")


def main():
    print("Cargando universo de genes (RNA-seq protein_coding)...")
    genes_orden, ensg_a_simbolo = cargar_universo_genes()
    genes_indice = {ensg: i for i, ensg in enumerate(genes_orden)}
    print(f"  {len(genes_orden)} genes")

    simbolo_a_ensg, simbolos_ambiguos, contadores_ambiguedad = construir_mapa_simbolo_ensg(ensg_a_simbolo)
    imprimir_contadores("mapa simbolo->ENSG (ambiguedad)", contadores_ambiguedad)

    print("Cargando mapa gen-proteina (BioMart)...")
    ensg_a_ensp, ensp_orden, contadores_biomart = cargar_mapa_ensg_ensp(genes_indice)
    ensp_indice = {ensp: i for i, ensp in enumerate(ensp_orden)}
    print(f"  {len(ensp_orden)} proteinas")
    imprimir_contadores("BioMart (gen-proteina)", contadores_biomart)

    print(f"Cargando interacciones proteina-proteina (STRING, umbral >= {UMBRAL_STRING})...")
    aristas_ppi, contadores_ppi = cargar_aristas_ppi(ensp_indice, UMBRAL_STRING)
    print(f"  {len(aristas_ppi)} interacciones")
    imprimir_contadores("STRING (proteina-proteina)", contadores_ppi)

    print("Cargando catalogo de vias Reactome...")
    vias_reactome_orden = cargar_catalogo_reactome()
    vias_reactome_indice = {v: i for i, v in enumerate(vias_reactome_orden)}
    print(f"  {len(vias_reactome_orden)} vias")

    print("Cargando aristas gen-via Reactome...")
    aristas_reactome, contadores_reactome = cargar_aristas_reactome(genes_indice, vias_reactome_indice)
    print(f"  {len(aristas_reactome)} aristas")
    imprimir_contadores("Reactome (gen-via)", contadores_reactome)

    print("Cargando catalogo de vias KEGG...")
    vias_kegg_orden = cargar_catalogo_kegg()
    vias_kegg_indice = {v: i for i, v in enumerate(vias_kegg_orden)}
    print(f"  {len(vias_kegg_orden)} vias")

    print("Cargando puente Entrez-simbolo (KEGG, solo CDS)...")
    entrez_a_simbolo, contadores_kegg_genes = cargar_mapa_entrez_simbolo()
    print(f"  {len(entrez_a_simbolo)} genes CDS con simbolo")
    imprimir_contadores("KEGG (Entrez-simbolo)", contadores_kegg_genes)

    print("Cargando aristas gen-via KEGG...")
    aristas_kegg, contadores_kegg_via = cargar_aristas_kegg(entrez_a_simbolo, simbolo_a_ensg, genes_indice, vias_kegg_indice)
    print(f"  {len(aristas_kegg)} aristas")
    imprimir_contadores("KEGG (gen-via)", contadores_kegg_via)

    print("Cargando aristas de regulacion transcripcional (TRRUST)...")
    aristas_regulacion, contadores_trrust = cargar_aristas_regulacion(simbolo_a_ensg, genes_indice)
    print(f"  {len(aristas_regulacion)} aristas")
    imprimir_contadores("TRRUST (regulacion)", contadores_trrust)

    print("Construyendo grafo heterogeneo (PyTorch Geometric)...")
    grafo = HeteroData()
    grafo["gen"].num_nodes = len(genes_orden)
    grafo["proteina"].num_nodes = len(ensp_orden)
    grafo["via_reactome"].num_nodes = len(vias_reactome_orden)
    grafo["via_kegg"].num_nodes = len(vias_kegg_orden)

    # Gen -> Proteina (codifica)
    origen = []
    destino = []
    for ensg, lista_ensp in ensg_a_ensp.items():
        for ensp in lista_ensp:
            origen.append(genes_indice[ensg])
            destino.append(ensp_indice[ensp])
    grafo["gen", "codifica", "proteina"].edge_index = torch.tensor([origen, destino], dtype=torch.long)

    # Proteina <-> Proteina (interactua), con combined_score como atributo
    origen = [a[0] for a in aristas_ppi]
    destino = [a[1] for a in aristas_ppi]
    scores = [a[2] for a in aristas_ppi]
    grafo["proteina", "interactua", "proteina"].edge_index = torch.tensor([origen, destino], dtype=torch.long)
    grafo["proteina", "interactua", "proteina"].edge_attr = torch.tensor(scores, dtype=torch.float)

    # Gen -> Via_Reactome (participa_en)
    origen = [genes_indice[ensg] for ensg, via_id in aristas_reactome]
    destino = [vias_reactome_indice[via_id] for ensg, via_id in aristas_reactome]
    grafo["gen", "participa_en", "via_reactome"].edge_index = torch.tensor([origen, destino], dtype=torch.long)

    # Gen -> Via_KEGG (participa_en)
    origen = [genes_indice[ensg] for ensg, via_id in aristas_kegg]
    destino = [vias_kegg_indice[via_id] for ensg, via_id in aristas_kegg]
    grafo["gen", "participa_en", "via_kegg"].edge_index = torch.tensor([origen, destino], dtype=torch.long)

    # Gen -> Gen (regula), dirigida TF -> diana, con modo como atributo
    origen = [genes_indice[ensg_tf] for ensg_tf, ensg_diana, modo in aristas_regulacion]
    destino = [genes_indice[ensg_diana] for ensg_tf, ensg_diana, modo in aristas_regulacion]
    codigos_modo = [MODO_REGULACION_A_CODIGO.get(modo, 0) for ensg_tf, ensg_diana, modo in aristas_regulacion]
    grafo["gen", "regula", "gen"].edge_index = torch.tensor([origen, destino], dtype=torch.long)
    grafo["gen", "regula", "gen"].edge_attr = torch.tensor(codigos_modo, dtype=torch.long)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(grafo, RUTA_SALIDA)

    print()
    print(f"Grafo guardado en {RUTA_SALIDA}")
    print(grafo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
