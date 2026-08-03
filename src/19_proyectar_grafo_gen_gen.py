import sys
from pathlib import Path
import torch

# Proyecta el PPI de STRING (grafo_heterogeneo.pt, aristas
# proteina-interactua-proteina) a un grafo gen-gen, porque la formula 2
# del TFM (Tanvir et al. 2024) define la atencion GAT sobre el
# vecindario N(i) del GEN i "en la red PPI", pero el grafo heterogeneo
# guarda el PPI a granularidad de PROTEINA (369.403 nodos proteina,
# muchas isoformas por gen via BioMart), no de gen.
#
# Metodo: invertir gen-codifica->proteina a proteina->gen; para cada
# arista proteina-interactua-proteina (p1, p2, score), proyectarla a
# (gen(p1), gen(p2), score). Se descartan:
#   - proteinas sin gen asociado en el universo (no deberia ocurrir,
#     BioMart ya se filtro a genes_indice en 13_construir_grafo.py, pero
#     se cuenta por seguridad)
#   - proteinas con MAS de un gen asociado (ambiguas): no hay forma de
#     saber a que gen atribuir la interaccion, se excluyen y se cuentan
#   - auto-bucles de isoforma (p1 y p2 codificadas por el MISMO gen): no
#     aportan informacion a nivel de gen
# Cuando varias parejas proteina-proteina proyectan al mismo par
# gen-gen (isoformas), nos quedamos con el score maximo (mismo criterio
# de "quedarse con la mejor evidencia disponible" que UMBRAL_STRING ya
# aplica a nivel de proteina). El grafo resultante se simetriza (cada
# par no dirigido -> 2 aristas dirigidas) para poder usarlo directamente
# como vecindario no dirigido en GATConv.

RUTA_GRAFO_HETEROGENEO = Path("data/processed/grafo_heterogeneo.pt")
RUTA_SALIDA = Path("data/processed/grafo_gen_gen_ppi.pt")


def construir_mapa_proteina_a_gen(grafo):
    origen, destino = grafo["gen", "codifica", "proteina"].edge_index
    proteina_a_genes = {}
    for gen_idx, proteina_idx in zip(origen.tolist(), destino.tolist()):
        proteina_a_genes.setdefault(proteina_idx, set()).add(gen_idx)

    proteina_a_gen = {}
    ambiguas = 0
    for proteina_idx, genes in proteina_a_genes.items():
        if len(genes) == 1:
            proteina_a_gen[proteina_idx] = next(iter(genes))
        else:
            ambiguas += 1

    contadores = {
        "proteinas_con_gen_unico": len(proteina_a_gen),
        "proteinas_ambiguas_excluidas": ambiguas,
    }
    return proteina_a_gen, contadores


def proyectar_aristas(grafo, proteina_a_gen):
    origen, destino = grafo["proteina", "interactua", "proteina"].edge_index
    scores = grafo["proteina", "interactua", "proteina"].edge_attr

    mejor_score = {}
    contadores = {
        "proteina_sin_gen_asociado": 0,
        "autobucle_isoforma_descartado": 0,
        "aristas_proyectadas_antes_dedup": 0,
    }
    for p1, p2, score in zip(origen.tolist(), destino.tolist(), scores.tolist()):
        g1 = proteina_a_gen.get(p1)
        g2 = proteina_a_gen.get(p2)
        if g1 is None or g2 is None:
            contadores["proteina_sin_gen_asociado"] += 1
            continue
        if g1 == g2:
            contadores["autobucle_isoforma_descartado"] += 1
            continue
        contadores["aristas_proyectadas_antes_dedup"] += 1
        clave = (g1, g2) if g1 < g2 else (g2, g1)
        if clave not in mejor_score or score > mejor_score[clave]:
            mejor_score[clave] = score

    return mejor_score, contadores


def main():
    print(f"Cargando grafo heterogeneo desde {RUTA_GRAFO_HETEROGENEO}...")
    grafo = torch.load(RUTA_GRAFO_HETEROGENEO, weights_only=False)
    n_genes = grafo["gen"].num_nodes
    print(f"  {n_genes} nodos gen")

    print("Invirtiendo gen-codifica->proteina a proteina->gen...")
    proteina_a_gen, contadores_mapa = construir_mapa_proteina_a_gen(grafo)
    print(f"  {contadores_mapa}")

    print("Proyectando aristas proteina-interactua->proteina a gen-gen...")
    mejor_score, contadores_proy = proyectar_aristas(grafo, proteina_a_gen)
    print(f"  {contadores_proy}")
    print(f"  pares gen-gen unicos (no dirigidos) tras dedup: {len(mejor_score)}")

    origen, destino, pesos = [], [], []
    for (g1, g2), score in mejor_score.items():
        origen.extend([g1, g2])
        destino.extend([g2, g1])
        pesos.extend([score, score])

    edge_index = torch.tensor([origen, destino], dtype=torch.long)
    edge_weight = torch.tensor(pesos, dtype=torch.float32)

    salida = {
        "num_nodes": n_genes,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "descripcion": (
            "Proyeccion gen-gen del PPI de STRING (proteina-interactua-proteina "
            "de grafo_heterogeneo.pt), via gen-codifica-proteina; mismo orden e "
            "indices de nodo 'gen' que grafo_heterogeneo.pt y que "
            "<COHORTE>_atributos_gen.pt (alineable por posicion)."
        ),
    }
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(salida, RUTA_SALIDA)

    print()
    print(f"Guardado: {RUTA_SALIDA}")
    print(f"Grafo gen-gen: {n_genes} nodos, {edge_index.shape[1]} aristas dirigidas "
          f"({len(mejor_score)} pares no dirigidos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
