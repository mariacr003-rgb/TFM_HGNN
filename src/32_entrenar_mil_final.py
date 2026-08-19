import sys, csv, importlib
from pathlib import Path
import torch

# Bloque 8 (MIL, Seccion 4.5 del TFM): ultimo paso, combinar el
# embedding z_WSI (Ilse et al. 2018, ya calculado por
# 27_procesar_wsi_paciente.py y guardado en
# data/processed/mil_wsi/<COHORTE>/<case_id>_mil.pt) con el embedding
# molecular de la Capa 3 (h_final, GAT+GCN, ya calculado por
# 25_entrenar_gat_gcn_final.py) segun la formula de la Seccion 4.5:
# vector conjunto = concat[h_final (32 dim) || z_WSI (2048 dim)] =
# 2080 dim por paciente.
#
# Reutiliza perdida_cox, indice_concordancia y cargar_supervivencia de
# 22_entrenar_gat_gcn.py (via importlib, mismo patron que 25 y 29), sin
# duplicarlas. Sin excepcion de estilo aqui (a diferencia de GAT/GCN/
# VGAE/MIL/DEC): la cabeza de riesgo es una unica capa lineal (mismo
# tipo que ModeloGATGCN.cabeza_riesgo en 21_modelo_gat_gcn.py), no hace
# falta una subclase de nn.Module.
#
# ADVERTENCIA DE SOBREAJUSTE (deliberada, documentada en el runlog): la
# muestra de MIL es de solo 10 pacientes por cohorte (Paso 50-51),
# frente a los ~168-185 del Bloque 6. La cabeza de riesgo tiene 2081
# parametros (2080 pesos + 1 sesgo) para 10 muestras - muy por encima
# de lo que cualquier regla practica de "muestras >> parametros"
# consideraria razonable. Se entrena UN UNICO modelo sobre TODOS los
# pacientes disponibles, sin pliegue de validacion (con n=10 no hay
# margen para un holdout con algun evento en cada lado): el C-index
# resultante es una medida de ajuste al propio conjunto de
# entrenamiento (como en 25_entrenar_gat_gcn_final.py), no de
# generalizacion, y con una muestra de este tamano probablemente
# refleja casi memorizacion perfecta, no una senal real. Se documenta
# explicitamente, sin intentar corregirlo con tecnicas fuera del
# alcance pedido (regularizacion, reduccion de dimensionalidad,
# arquitectura distinta a la formula de la Seccion 4.5).

sys.path.insert(0, str(Path(__file__).resolve().parent))
_mod22 = importlib.import_module("22_entrenar_gat_gcn")
cargar_supervivencia = _mod22.cargar_supervivencia
perdida_cox = _mod22.perdida_cox
indice_concordancia = _mod22.indice_concordancia

N_EPOCAS = 20
LR = 1e-3
SEMILLA = 0

# C-index solo molecular ya documentado (Bloque 6, Paso 40, validacion
# cruzada 5-fold sobre ~168-185 pacientes) - para la comparacion
# directa que pide este paso.
C_INDEX_SOLO_MOLECULAR = {
    "BRCA": 0.6255,
    "LUAD": 0.5329,
    "LUSC": 0.4514,
    "COAD": 0.4599,
    "KIRC": 0.4928,
}


def cargar_vectores_conjuntos(cohorte):
    # Para cada paciente de <COHORTE>_pacientes_mil.tsv con z_wsi ya
    # calculado, busca su fila de h_final en el modelo GAT+GCN final
    # (mismo case_id) y concatena [h_final (32) || z_wsi (2048)].
    d_gat = torch.load(f"data/processed/{cohorte}_modelo_gat_gcn_final.pt",
                        map_location="cpu", weights_only=False)
    pacientes_gat = d_gat["pacientes"]
    h_final = d_gat["h_final"]
    indice_por_caso = {c: i for i, c in enumerate(pacientes_gat)}

    with open(f"data/processed/{cohorte}_pacientes_mil.tsv", newline="", encoding="utf-8") as f:
        mil_case_ids = [fila["case_id"] for fila in csv.DictReader(f, delimiter="\t")]

    case_ids, vectores = [], []
    for case_id in mil_case_ids:
        ruta_mil = Path(f"data/processed/mil_wsi/{cohorte}/{case_id}_mil.pt")
        if not ruta_mil.exists():
            print(f"  AVISO: {case_id} no tiene z_wsi calculado, se omite")
            continue
        if case_id not in indice_por_caso:
            print(f"  AVISO: {case_id} no esta en el modelo GAT+GCN final, se omite")
            continue
        d_mil = torch.load(ruta_mil, map_location="cpu", weights_only=False)
        z_wsi = d_mil["z_wsi"]
        h = h_final[indice_por_caso[case_id]]
        vector = torch.cat([h, z_wsi], dim=0)
        case_ids.append(case_id)
        vectores.append(vector)
    x = torch.stack(vectores) if vectores else torch.empty(0, 32 + 2048)
    return case_ids, x


def main(cohorte, ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cohorte: {cohorte}")
    print("Cargando y concatenando h_final (GAT+GCN) + z_wsi (MIL)...")
    case_ids, x = cargar_vectores_conjuntos(cohorte)
    print(f"  {len(case_ids)} pacientes con vector conjunto: {tuple(x.shape)}")

    print("Cruzando con supervivencia (clinical_valid.tsv)...")
    indices_validos, tiempo, evento = cargar_supervivencia(cohorte, case_ids)
    x = x[indices_validos]
    case_ids_validos = [case_ids[i] for i in indices_validos]
    n_pacientes = x.shape[0]
    n_eventos = int(evento.sum().item())
    print(f"  {n_pacientes} pacientes con supervivencia utilizable ({n_eventos} eventos)")

    n_parametros_cabeza = x.shape[1] + 1 if n_pacientes > 0 else 0
    print(f"  ADVERTENCIA DE SOBREAJUSTE: {n_parametros_cabeza} parametros en la cabeza de "
          f"riesgo (dim={x.shape[1] if n_pacientes else 0}+sesgo) para {n_pacientes} pacientes - "
          f"ver runlog para la discusion completa.")

    resultado = {
        "cohorte": cohorte,
        "case_ids": case_ids_validos,
        "n_pacientes": n_pacientes,
        "n_eventos": n_eventos,
        "n_parametros_cabeza_riesgo": n_parametros_cabeza,
        "c_index_solo_molecular_bloque6": C_INDEX_SOLO_MOLECULAR[cohorte],
    }

    if n_eventos == 0:
        print("  SIN EVENTOS: la perdida de Cox esta indefinida con 0 eventos observados "
              "(ver perdida_cox en 22_entrenar_gat_gcn.py) - NO se entrena, NO hay C-index "
              "MIL+molecular para esta cohorte con esta muestra de 10 pacientes.")
        resultado.update({
            "estado": "SIN_EVENTOS",
            "c_index_mil_molecular": None,
            "n_pares_comparables": 0,
        })
    else:
        torch.manual_seed(SEMILLA)
        cabeza_riesgo = torch.nn.Linear(x.shape[1], 1)
        optimizador = torch.optim.Adam(cabeza_riesgo.parameters(), lr=LR)

        print(f"Entrenando {N_EPOCAS} epocas sobre los {n_pacientes} pacientes "
              "(sin pliegue de validacion - ver advertencia de sobreajuste arriba)...")
        for epoca in range(N_EPOCAS):
            cabeza_riesgo.train()
            optimizador.zero_grad()
            riesgo = cabeza_riesgo(x).squeeze(-1)
            perdida = perdida_cox(riesgo, tiempo, evento)
            perdida.backward()
            optimizador.step()
            print(f"  epoca {epoca + 1}/{N_EPOCAS}: perdida_cox={perdida.item():.4f}")

        cabeza_riesgo.eval()
        with torch.no_grad():
            riesgo_final = cabeza_riesgo(x).squeeze(-1)
            c_index = indice_concordancia(riesgo_final, tiempo, evento)

        # Numero de pares comparables real (informe de fiabilidad
        # estadistica, no solo el valor del C-index): con pocos
        # eventos, el C-index se calcula sobre muy pocos pares.
        tiempo_l, evento_l = tiempo.tolist(), evento.tolist()
        n_pares = sum(1 for i in range(n_pacientes) for j in range(n_pacientes)
                      if evento_l[i] == 1 and tiempo_l[j] > tiempo_l[i])

        resultado.update({
            "estado": "ENTRENADO",
            "c_index_mil_molecular": c_index,
            "n_pares_comparables": n_pares,
            "state_dict_cabeza_riesgo": cabeza_riesgo.state_dict(),
        })
        print()
        print(f"C-index MIL+molecular (sobre el propio conjunto de entrenamiento, "
              f"NO es generalizacion): {c_index:.4f}")
        print(f"  calculado sobre {n_pares} pares comparables (de un maximo teorico de "
              f"{n_pacientes * (n_pacientes - 1)} pares totales)")

    torch.save(resultado, ruta_salida.with_suffix(".pt"))

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(f"Cohorte: {cohorte}\n")
        f.write(f"Pacientes con vector conjunto MIL+molecular (32+2048=2080 dim): {n_pacientes}\n")
        f.write(f"Eventos observados: {n_eventos}\n")
        f.write(f"Parametros de la cabeza de riesgo (lineal, 2080 dim + sesgo): {n_parametros_cabeza}\n")
        f.write(f"Estado: {resultado['estado']}\n")
        if resultado["estado"] == "SIN_EVENTOS":
            f.write("C-index MIL+molecular: NO CALCULABLE (0 eventos, perdida de Cox indefinida)\n")
        else:
            f.write(f"C-index MIL+molecular (entrenamiento, NO generalizacion): {resultado['c_index_mil_molecular']:.4f}\n")
            f.write(f"Pares comparables usados en el C-index: {resultado['n_pares_comparables']}\n")
        f.write(f"C-index solo molecular (Bloque 6, validacion cruzada 5-fold, ~168-185 pacientes): "
                f"{C_INDEX_SOLO_MOLECULAR[cohorte]:.4f}\n")
        f.write("ADVERTENCIA: los dos C-index NO son directamente comparables - protocolos de "
                "evaluacion distintos (validacion cruzada sobre ~200 pacientes vs. C-index de "
                "entrenamiento sobre 10 pacientes, sin holdout). Ver runlog para la discusion "
                "completa de fiabilidad estadistica.\n")

    print(f"Guardado: {ruta_salida} (resumen legible) y {ruta_salida.with_suffix('.pt')} (con state_dict)")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_salida = argumentos[1] if len(argumentos) > 1 else f"data/processed/{cohorte}_mil_final_metricas.txt"
    sys.exit(main(cohorte, ruta_salida))
