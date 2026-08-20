import csv, sys, random, importlib
from pathlib import Path
import torch

# Entrena y evalua, para UNA cohorte, el modelo GAT+GCN definido en
# 21_modelo_gat_gcn.py con validacion cruzada K-fold estratificada por
# evento (ver runlog del Bloque 6: con ~200 pacientes y pocos eventos
# observados por cohorte, K-fold es mas robusto que un hold-out unico).
# Un modelo INDEPENDIENTE por cohorte (no pooled entre cohortes): mismo
# criterio ya usado en el resto del pipeline (cada cohorte con su propio
# <COHORTE>_atributos_gen.pt), y coherente con el protocolo de
# evaluacion por cohorte de la Seccion 4.8 del TFM.
#
# 21_modelo_gat_gcn.py se numera como los demas scripts de src/ (sin
# guion bajo al principio), asi que no es un identificador de modulo
# valido para "import 21_modelo_gat_gcn"; se carga con importlib
# (verificado que funciona igual que con el resto de scripts numerados).
#
# Import de ModeloGATGCN DIFERIDO a entrenar_un_pliegue() (unico sitio
# donde se usa), en vez de a nivel de modulo: 21_modelo_gat_gcn.py
# importa torch_geometric, una dependencia pesada que NO hace falta
# para reutilizar cargar_supervivencia()/construir_pliegues()/
# indice_concordancia() desde otros scripts (25, 29, 32, 34) - varios
# de ellos corren en entornos sin torch_geometric instalado (ver
# src/34_baseline_svm_rf.py, venv aislado sin PyG). Cambio puramente de
# ubicacion del import, sin tocar su comportamiento cuando se ejecuta
# este script normalmente (con torch_geometric disponible).

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUTA_GRAFO_GEN_GEN = Path("data/processed/grafo_gen_gen_ppi.pt")


def cargar_grafo_gen_gen():
    datos = torch.load(RUTA_GRAFO_GEN_GEN, weights_only=False)
    return datos["edge_index"], datos["num_nodes"]


def cargar_atributos_normalizados(cohorte):
    ruta = Path("data/processed") / f"{cohorte}_atributos_gen_norm.pt"
    return torch.load(ruta, weights_only=False)


def cargar_supervivencia(cohorte, pacientes):
    # Cruza la lista de pacientes del tensor de atributos con
    # <COHORTE>_clinical_valid.tsv por case_id. tiempo = days_to_death si
    # vital_status es Dead (evento=1), days_to_last_follow_up si es
    # Alive (evento=0). Pacientes sin fila clinica, con vital_status
    # distinto de Alive/Dead, o sin el campo de tiempo correspondiente,
    # se excluyen (devueltos en indices_validos).
    ruta_clinico = Path("data/processed") / f"{cohorte}_clinical_valid.tsv"
    datos_por_caso = {}
    with ruta_clinico.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            datos_por_caso[fila["case_id"]] = fila

    indices_validos, tiempos, eventos = [], [], []
    for i, caso in enumerate(pacientes):
        fila = datos_por_caso.get(caso)
        if fila is None:
            continue
        if fila["vital_status"] == "Alive":
            tiempo_str, evento = fila["days_to_last_follow_up"], 0
        elif fila["vital_status"] == "Dead":
            tiempo_str, evento = fila["days_to_death"], 1
        else:
            continue
        if not tiempo_str:
            continue
        indices_validos.append(i)
        tiempos.append(float(tiempo_str))
        eventos.append(evento)

    tiempo = torch.tensor(tiempos, dtype=torch.float32)
    evento = torch.tensor(eventos, dtype=torch.long)
    return indices_validos, tiempo, evento


def perdida_cox(riesgo, tiempo, evento):
    # Formula estandar de Cox (log-verosimilitud parcial negativa):
    # L_Cox = -sum_{i: delta_i=1} [ y_i - log(sum_{j: t_j>=t_i} exp(y_j)) ]
    # Con tiempo ordenado de mayor a menor, el conjunto en riesgo de la
    # posicion k (todo j con t_j >= tiempo_k) es exactamente
    # riesgo_ordenado[0..k], asi que logcumsumexp da la suma interior
    # directamente, sin bucles.
    if evento.sum() == 0:
        return None
    orden = torch.argsort(tiempo, descending=True)
    riesgo_ordenado = riesgo[orden]
    evento_ordenado = evento[orden]
    log_riesgo_acumulado = torch.logcumsumexp(riesgo_ordenado, dim=0)
    log_verosimilitud = riesgo_ordenado - log_riesgo_acumulado
    return -log_verosimilitud[evento_ordenado.bool()].sum()


def indice_concordancia(riesgo, tiempo, evento):
    # C-index estandar: entre todos los pares (i, j) comparables (i con
    # evento observado y t_i < t_j), proporcion en la que el riesgo
    # predicho respeta el orden real (riesgo_i > riesgo_j); empates de
    # riesgo cuentan 0.5. O(n^2), trivial para ~200 pacientes.
    riesgo_l = riesgo.detach().tolist()
    tiempo_l = tiempo.tolist()
    evento_l = evento.tolist()
    n = len(tiempo_l)
    concordantes, comparables = 0.0, 0
    for i in range(n):
        if evento_l[i] != 1:
            continue
        for j in range(n):
            if tiempo_l[j] <= tiempo_l[i]:
                continue
            comparables += 1
            if riesgo_l[i] > riesgo_l[j]:
                concordantes += 1.0
            elif riesgo_l[i] == riesgo_l[j]:
                concordantes += 0.5
    return concordantes / comparables if comparables > 0 else float("nan")


def construir_pliegues(n_pacientes, eventos, n_pliegues, semilla):
    # K-fold estratificado por evento: reparte por separado los indices
    # con evento=1 y evento=0 en round-robin, para no dejar ningun
    # pliegue sin eventos observados (la tasa de eventos en TCGA suele
    # ser baja, ej. 17/181 en BRCA).
    generador = random.Random(semilla)
    indices_evento = [i for i in range(n_pacientes) if eventos[i] == 1]
    indices_censura = [i for i in range(n_pacientes) if eventos[i] == 0]
    generador.shuffle(indices_evento)
    generador.shuffle(indices_censura)

    pliegues = [[] for _ in range(n_pliegues)]
    for i, idx in enumerate(indices_evento):
        pliegues[i % n_pliegues].append(idx)
    for i, idx in enumerate(indices_censura):
        pliegues[i % n_pliegues].append(idx)
    return pliegues


def entrenar_un_pliegue(x, edge_index_gen_gen, tiempo, evento, indices_train, indices_val, n_epocas, k_vecinos):
    ModeloGATGCN = importlib.import_module("21_modelo_gat_gcn").ModeloGATGCN
    modelo = ModeloGATGCN(k_vecinos=k_vecinos)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    x_train = x[indices_train]
    tiempo_train = tiempo[indices_train]
    evento_train = evento[indices_train]

    for epoca in range(n_epocas):
        modelo.train()
        optimizador.zero_grad()
        riesgo_train, _h_final_train = modelo(x_train, edge_index_gen_gen)
        perdida = perdida_cox(riesgo_train, tiempo_train, evento_train)
        if perdida is None:
            print(f"    epoca {epoca + 1}/{n_epocas}: sin eventos en el pliegue de entrenamiento, se omite")
            continue
        perdida.backward()
        optimizador.step()
        print(f"    epoca {epoca + 1}/{n_epocas}: perdida_cox={perdida.item():.4f}")

    modelo.eval()
    with torch.no_grad():
        riesgo_val, _h_final_val = modelo(x[indices_val], edge_index_gen_gen)
        c_index = indice_concordancia(riesgo_val, tiempo[indices_val], evento[indices_val])
    return c_index


def main(cohorte, ruta_salida, n_epocas, n_pliegues, n_pacientes_max, k_vecinos):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"  {len(indices_validos)} / {len(pacientes)} pacientes con supervivencia utilizable "
          f"({int(evento.sum().item())} eventos observados)")

    x = x[indices_validos]

    if n_pacientes_max and n_pacientes_max < x.shape[0]:
        print(f"[SUBMUESTREO] limitando a los primeros {n_pacientes_max} pacientes (uso: smoke test)")
        x = x[:n_pacientes_max]
        tiempo = tiempo[:n_pacientes_max]
        evento = evento[:n_pacientes_max]

    n_pacientes = x.shape[0]
    print(f"Pacientes finales para entrenamiento/evaluacion: {n_pacientes} "
          f"({int(evento.sum().item())} eventos)")

    pliegues = construir_pliegues(n_pacientes, evento.tolist(), n_pliegues, semilla=0)

    c_indices = []
    for k in range(n_pliegues):
        indices_val = pliegues[k]
        indices_train = [i for f in range(n_pliegues) if f != k for i in pliegues[f]]
        n_eventos_val = sum(evento[i].item() for i in indices_val)
        print(f"Pliegue {k + 1}/{n_pliegues}: {len(indices_train)} train, "
              f"{len(indices_val)} val ({n_eventos_val} eventos en val)")
        c_index = entrenar_un_pliegue(
            x, edge_index_gen_gen, tiempo, evento, indices_train, indices_val, n_epocas, k_vecinos
        )
        texto_c = f"{c_index:.4f}" if c_index == c_index else "no evaluable (sin pares comparables)"
        print(f"  C-index pliegue {k + 1}: {texto_c}")
        c_indices.append(c_index)

    c_indices_validos = [c for c in c_indices if c == c]
    media = sum(c_indices_validos) / len(c_indices_validos) if c_indices_validos else float("nan")
    if len(c_indices_validos) > 1:
        varianza = sum((c - media) ** 2 for c in c_indices_validos) / len(c_indices_validos)
    else:
        varianza = 0.0
    desviacion = varianza ** 0.5

    lineas = [
        f"Cohorte: {cohorte}",
        f"Pacientes usados: {n_pacientes} (de {len(pacientes)} totales, "
        f"{len(indices_validos)} con supervivencia utilizable)",
        f"Eventos observados: {int(evento.sum().item())}",
        f"Pliegues: {n_pliegues}, epocas por pliegue: {n_epocas}, k vecinos (kNN pacientes): {k_vecinos}",
        "",
        "C-index por pliegue:",
    ]
    for i, c in enumerate(c_indices):
        lineas.append(f"  pliegue {i + 1}: {c:.4f}" if c == c else f"  pliegue {i + 1}: no evaluable")
    lineas.append("")
    lineas.append(f"C-index medio +/- desviacion estandar: {media:.4f} +/- {desviacion:.4f}")

    texto_final = "\n".join(lineas)
    ruta_salida.write_text(texto_final, encoding="utf-8")

    print()
    print(texto_final)
    print(f"\nGuardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_salida = argumentos[1]
    n_epocas = int(argumentos[2]) if len(argumentos) > 2 else 50
    n_pliegues = int(argumentos[3]) if len(argumentos) > 3 else 5
    n_pacientes_max = int(argumentos[4]) if len(argumentos) > 4 else 0
    k_vecinos = int(argumentos[5]) if len(argumentos) > 5 else 20
    sys.exit(main(cohorte, ruta_salida, n_epocas, n_pliegues, n_pacientes_max, k_vecinos))
