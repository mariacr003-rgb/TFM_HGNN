import sys, importlib
from pathlib import Path
import numpy as np
import torch

# Baseline SVM/RF de una sola capa omica (Seccion 1.3 del TFM, objetivo
# declarado pero nunca ejecutado hasta este paso - ver diagnostico de
# viabilidad previo, sin ajustes al confirmarse que la Seccion 4.8 no
# fija ni la capa ni el K de reduccion de dimensionalidad).
#
# Capa omica: RNA-seq (canal 0, "rnaseq_tpm", ya imputado y normalizado
# por 20_preprocesar_atributos_gen.py) de <COHORTE>_atributos_gen_norm.pt
# - el mismo tensor que usa GAT+GCN, sin descargar ni procesar nada
# nuevo.
#
# Reduccion de dimensionalidad: top-K genes por varianza (mismo criterio
# ya usado en VGAE, Paso 43, aunque con un K distinto - alli 2500 por
# motivo de arquitectura del autoencoder, aqui 1000 por el ratio
# features/muestras de un modelo clasico, ver diagnostico de
# viabilidad).
#
# Modelos: RandomSurvivalForest y FastSurvivalSVM (kernel lineal) de
# scikit-survival - instalado en un venv AISLADO
# (tfm_entorno/venv_baseline_svm_rf), no en venv_pytorch_wsl, porque no
# necesita torch/PyG para nada y asi se evita cualquier riesgo de
# conflicto de version con el entorno ya validado de GAT+GCN/VGAE/MIL/DEC.
# Este script SI importa torch, pero solo para leer los .pt del
# proyecto (I/O) - ningun calculo usa torch, todo es numpy/sksurv.
#
# Reutilizado de 22_entrenar_gat_gcn.py (via importlib, mismo patron que
# 25/29/32): cargar_supervivencia(), construir_pliegues() e
# indice_concordancia() - sin reimplementar ninguna. El C-index se
# calcula con la MISMA funcion que el Bloque 6 (no la propia de
# scikit-survival), para que la comparacion sea valida.

sys.path.insert(0, str(Path(__file__).resolve().parent))
_mod22 = importlib.import_module("22_entrenar_gat_gcn")
cargar_supervivencia = _mod22.cargar_supervivencia
construir_pliegues = _mod22.construir_pliegues
indice_concordancia = _mod22.indice_concordancia

K_GENES = 1000
N_PLIEGUES = 5
SEMILLA = 0
N_ESTIMATORS_RSF = 300

# C-index solo molecular ya documentado (Bloque 6, Paso 40) - para la
# comparacion directa que pide este paso.
C_INDEX_GAT_GCN = {
    "BRCA": 0.6255,
    "LUAD": 0.5329,
    "LUSC": 0.4514,
    "COAD": 0.4599,
    "KIRC": 0.4928,
}


def cargar_rnaseq(cohorte):
    d = torch.load(f"data/processed/{cohorte}_atributos_gen_norm.pt", map_location="cpu", weights_only=False)
    x_rnaseq = d["x"][:, :, 0].numpy().astype(np.float64)  # canal 0 = rnaseq_tpm
    pacientes = d["pacientes"]
    return x_rnaseq, pacientes


def seleccionar_top_k_varianza(x, k):
    varianzas = x.var(axis=0)
    indices_top_k = np.argsort(varianzas)[::-1][:k]
    return x[:, indices_top_k]


def indice_concordancia_np(riesgo_np, tiempo_np, evento_np):
    # indice_concordancia() (22_entrenar_gat_gcn.py) espera tensores de
    # torch (usa .detach() en riesgo) - conversion sin reimplementar
    # la logica del C-index.
    return indice_concordancia(
        torch.from_numpy(np.asarray(riesgo_np, dtype=np.float64)),
        torch.from_numpy(tiempo_np),
        torch.from_numpy(evento_np.astype(np.int64)),
    )


def main(cohorte, modelos, ruta_salida):
    from sksurv.util import Surv

    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cohorte: {cohorte}")
    print("Cargando RNA-seq (canal 0 de atributos_gen_norm.pt, ya imputado y normalizado)...")
    x_rnaseq, pacientes = cargar_rnaseq(cohorte)
    print(f"  {x_rnaseq.shape[0]} pacientes, {x_rnaseq.shape[1]} genes")

    print("Cruzando con supervivencia (clinical_valid.tsv)...")
    indices_validos, tiempo, evento = cargar_supervivencia(cohorte, pacientes)
    x = x_rnaseq[indices_validos]
    tiempo_np = tiempo.numpy().astype(np.float64)
    evento_np = evento.numpy().astype(np.int64)
    n_pacientes = x.shape[0]
    print(f"  {n_pacientes} pacientes con supervivencia utilizable ({int(evento_np.sum())} eventos)")

    print(f"Reduciendo a los {K_GENES} genes de mayor varianza...")
    x_topk = seleccionar_top_k_varianza(x, K_GENES)
    print(f"  x: {x_topk.shape}")

    pliegues = construir_pliegues(n_pacientes, evento.tolist(), N_PLIEGUES, SEMILLA)
    print(f"Validacion cruzada {N_PLIEGUES}-fold estratificada por evento (mismo protocolo que GAT+GCN)...")

    resultados_por_modelo = {m: [] for m in modelos}
    for i in range(N_PLIEGUES):
        indices_val = pliegues[i]
        indices_train = [idx for j, p in enumerate(pliegues) if j != i for idx in p]
        n_eventos_val = int(evento_np[indices_val].sum())
        print(f"  Pliegue {i + 1}/{N_PLIEGUES}: {len(indices_train)} train, {len(indices_val)} val "
              f"({n_eventos_val} eventos en val)")

        x_train, x_val = x_topk[indices_train], x_topk[indices_val]
        tiempo_val, evento_val = tiempo_np[indices_val], evento_np[indices_val]

        if evento_np[indices_train].sum() == 0:
            print("    sin eventos en el pliegue de entrenamiento, se omite (mismo criterio que GAT+GCN)")
            continue

        y_train = Surv.from_arrays(evento_np[indices_train].astype(bool), tiempo_np[indices_train])

        if "rsf" in modelos:
            from sksurv.ensemble import RandomSurvivalForest
            rsf = RandomSurvivalForest(n_estimators=N_ESTIMATORS_RSF, random_state=SEMILLA, n_jobs=-1)
            rsf.fit(x_train, y_train)
            riesgo_val = rsf.predict(x_val)
            c_index = indice_concordancia_np(riesgo_val, tiempo_val, evento_val)
            resultados_por_modelo["rsf"].append(c_index)
            print(f"    RandomSurvivalForest: C-index={c_index:.4f}")

        if "svm" in modelos:
            from sksurv.svm import FastSurvivalSVM
            svm = FastSurvivalSVM(random_state=SEMILLA, max_iter=1000)
            svm.fit(x_train, y_train)
            riesgo_val = svm.predict(x_val)
            c_index = indice_concordancia_np(riesgo_val, tiempo_val, evento_val)
            resultados_por_modelo["svm"].append(c_index)
            print(f"    FastSurvivalSVM: C-index={c_index:.4f}")

    print()
    resultado = {"cohorte": cohorte, "k_genes": K_GENES, "n_pliegues": N_PLIEGUES,
                 "c_index_gat_gcn_bloque6": C_INDEX_GAT_GCN[cohorte]}
    lineas_txt = [f"Cohorte: {cohorte}", f"Genes (top-K por varianza): {K_GENES}",
                  f"Pliegues: {N_PLIEGUES}"]
    for m in modelos:
        valores = resultados_por_modelo[m]
        nombre = {"rsf": "RandomSurvivalForest", "svm": "FastSurvivalSVM"}[m]
        if valores:
            media = float(np.mean(valores))
            desv = float(np.std(valores))
            print(f"{nombre}: C-index medio = {media:.4f} +/- {desv:.4f} (pliegues: {[f'{v:.4f}' for v in valores]})")
            resultado[f"c_index_{m}_medio"] = media
            resultado[f"c_index_{m}_std"] = desv
            resultado[f"c_index_{m}_pliegues"] = valores
            lineas_txt.append(f"{nombre}: C-index medio = {media:.4f} +/- {desv:.4f}")
            lineas_txt.append(f"  pliegues: {[round(v, 4) for v in valores]}")
        else:
            print(f"{nombre}: SIN RESULTADO (todos los pliegues de entrenamiento sin eventos)")
            resultado[f"c_index_{m}_medio"] = None
            lineas_txt.append(f"{nombre}: SIN RESULTADO (todos los pliegues de entrenamiento sin eventos)")
    lineas_txt.append(f"C-index GAT+GCN (Bloque 6, referencia): {C_INDEX_GAT_GCN[cohorte]:.4f}")

    torch.save(resultado, ruta_salida.with_suffix(".pt"))
    ruta_salida.write_text("\n".join(lineas_txt) + "\n", encoding="utf-8")
    print(f"Guardado: {ruta_salida} y {ruta_salida.with_suffix('.pt')}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    modelos = argumentos[1].split(",") if len(argumentos) > 1 else ["rsf", "svm"]
    ruta_salida = argumentos[2] if len(argumentos) > 2 else f"data/processed/{cohorte}_baseline_svm_rf_metricas.txt"
    sys.exit(main(cohorte, modelos, ruta_salida))
