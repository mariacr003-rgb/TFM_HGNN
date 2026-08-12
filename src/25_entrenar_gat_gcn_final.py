import sys, importlib
from pathlib import Path
import torch

# Bloque 8 (MIL): PRERREQUISITO independiente de MIL en si. El
# entrenamiento de GAT+GCN del Bloque 6 (22_entrenar_gat_gcn.py) solo
# hace validacion cruzada (un modelo nuevo POR PLIEGUE, descartado tras
# calcular su C-index) - nunca guarda un modelo final utilizable, asi
# que el "embedding molecular de la Capa 3" que pide la formula de MIL
# (Seccion 4.5) no existe todavia en disco.
#
# Este script entrena UN UNICO modelo sobre TODOS los pacientes con
# supervivencia utilizable de una cohorte (sin pliegues: el objetivo
# aqui no es evaluar generalizacion -eso ya se hizo en el Bloque 6,
# Paso 36-, sino producir un modelo final del que extraer h_final por
# paciente) y guarda su state_dict con torch.save().
#
# Reutiliza (via importlib, mismo patron que 22 usa para importar 21)
# las funciones de carga de datos y la perdida de Cox ya definidas en
# 22_entrenar_gat_gcn.py, sin duplicarlas.

sys.path.insert(0, str(Path(__file__).resolve().parent))
_mod21 = importlib.import_module("21_modelo_gat_gcn")
ModeloGATGCN = _mod21.ModeloGATGCN
_mod22 = importlib.import_module("22_entrenar_gat_gcn")
cargar_grafo_gen_gen = _mod22.cargar_grafo_gen_gen
cargar_atributos_normalizados = _mod22.cargar_atributos_normalizados
cargar_supervivencia = _mod22.cargar_supervivencia
perdida_cox = _mod22.perdida_cox
indice_concordancia = _mod22.indice_concordancia


def main(cohorte, ruta_salida_modelo, n_epocas, k_vecinos):
    ruta_salida_modelo = Path(ruta_salida_modelo)
    ruta_salida_modelo.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cohorte: {cohorte}")
    print("Cargando grafo gen-gen (proyeccion PPI STRING, ver 19_proyectar_grafo_gen_gen.py)...")
    edge_index_gen_gen, n_genes_grafo = cargar_grafo_gen_gen()
    print(f"  {n_genes_grafo} genes, {edge_index_gen_gen.shape[1]} aristas dirigidas")

    print("Cargando atributos por gen normalizados (ver 20_preprocesar_atributos_gen.py)...")
    datos_atributos = cargar_atributos_normalizados(cohorte)
    x = datos_atributos["x"]
    pacientes = datos_atributos["pacientes"]
    if x.shape[1] != n_genes_grafo:
        raise ValueError("El numero de genes del tensor de atributos no coincide con el grafo gen-gen.")

    print("Cruzando con supervivencia (clinical_valid.tsv)...")
    indices_validos, tiempo, evento = cargar_supervivencia(cohorte, pacientes)
    x = x[indices_validos]
    pacientes_validos = [pacientes[i] for i in indices_validos]
    n_pacientes = x.shape[0]
    print(f"  {n_pacientes} pacientes con supervivencia utilizable ({int(evento.sum().item())} eventos)")
    print("  ENTRENAMIENTO SOBRE EL CONJUNTO COMPLETO, sin pliegues: el objetivo de este script es "
          "un modelo final para extraer embeddings, no evaluar generalizacion (ya evaluada en el Paso 36).")

    modelo = ModeloGATGCN(k_vecinos=k_vecinos)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    print(f"Entrenando {n_epocas} epocas sobre los {n_pacientes} pacientes...")
    for epoca in range(n_epocas):
        modelo.train()
        optimizador.zero_grad()
        riesgo, _h_final = modelo(x, edge_index_gen_gen)
        perdida = perdida_cox(riesgo, tiempo, evento)
        perdida.backward()
        optimizador.step()
        print(f"  epoca {epoca + 1}/{n_epocas}: perdida_cox={perdida.item():.4f}")

    modelo.eval()
    with torch.no_grad():
        riesgo_final, h_final = modelo(x, edge_index_gen_gen)
        # C-index sobre el propio conjunto de entrenamiento: NO es una
        # metrica de generalizacion (no hay pliegue de validacion aqui,
        # ver nota arriba), solo una comprobacion de cordura de que el
        # modelo aprendio algo coherente antes de guardarlo.
        c_index_entrenamiento = indice_concordancia(riesgo_final, tiempo, evento)

    torch.save({
        "cohorte": cohorte,
        "state_dict": modelo.state_dict(),
        "pacientes": pacientes_validos,
        "h_final": h_final,  # [n_pacientes, dim_salida_gcn=32], embedding Capa 3 ya calculado
        "n_epocas": n_epocas,
        "k_vecinos": k_vecinos,
        "c_index_entrenamiento": c_index_entrenamiento,
    }, ruta_salida_modelo)

    print()
    print(f"C-index sobre el conjunto de entrenamiento (NO es generalizacion): {c_index_entrenamiento:.4f}")
    print(f"h_final: {tuple(h_final.shape)}")
    print(f"Guardado: {ruta_salida_modelo}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_salida_modelo = argumentos[1]
    n_epocas = int(argumentos[2]) if len(argumentos) > 2 else 20
    k_vecinos = int(argumentos[3]) if len(argumentos) > 3 else 20
    sys.exit(main(cohorte, ruta_salida_modelo, n_epocas, k_vecinos))
