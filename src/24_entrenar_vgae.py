import sys, importlib
from pathlib import Path
import torch

# Bloque 7: pre-entrena el VGAE (23_modelo_vgae.py) para UNA cohorte,
# de forma INDEPENDIENTE del GAT+GCN (Bloque 6) y de las demas
# cohortes (mismo criterio ya usado en 22_entrenar_gat_gcn.py: un
# modelo por cohorte, no pooled). Pre-entrenamiento NO SUPERVISADO
# (Kipf y Welling 2016): no usa la supervivencia clinica en absoluto,
# solo reconstruye los atributos por gen.
#
# 21_modelo_gat_gcn.py y 23_modelo_vgae.py se numeran como el resto de
# scripts de src/ (sin guion bajo al principio), por lo que no son
# nombres de modulo validos para "import 21_modelo_gat_gcn"; se cargan
# con importlib, mismo patron ya usado en 22_entrenar_gat_gcn.py.

sys.path.insert(0, str(Path(__file__).resolve().parent))
construir_grafo_knn_pacientes = importlib.import_module("21_modelo_gat_gcn").construir_grafo_knn_pacientes
_mod_vgae = importlib.import_module("23_modelo_vgae")
ModeloVGAE = _mod_vgae.ModeloVGAE
perdida_elbo = _mod_vgae.perdida_elbo


def cargar_atributos_crudos(cohorte):
    # Tensor crudo (18_construir_atributos_gen.py): x con NaN reales
    # en las posiciones invalidas, mascara_valido aparte. Aqui solo se
    # usa la mascara (el valor imputado/normalizado se toma del
    # tensor de 20_preprocesar_atributos_gen.py, ver
    # cargar_atributos_normalizados).
    ruta = Path("data/processed") / f"{cohorte}_atributos_gen.pt"
    return torch.load(ruta, weights_only=False)


def cargar_atributos_normalizados(cohorte):
    ruta = Path("data/processed") / f"{cohorte}_atributos_gen_norm.pt"
    return torch.load(ruta, weights_only=False)


def construir_mascara_sintetica(mascara_valido, fraccion, semilla):
    # Enmascara artificialmente 'fraccion' de las celdas YA VALIDAS,
    # ESTRATIFICADO POR CANAL (RNA-seq, CNV y metilacion tienen
    # cobertura real muy distinta - 100%, 99.7%, 60.4% - estratificar
    # evita que la retencion se concentre desproporcionadamente en un
    # canal). Devuelve:
    #   mascara_entrenamiento: validas MENOS las retenidas (lo que ve
    #     el modelo, tanto en la perdida como en la mascara de entrada)
    #   mascara_retenida: las retenidas, con valor real conocido,
    #     usadas UNICAMENTE para evaluar la calidad de imputacion
    #     (nunca entran en la perdida de entrenamiento)
    generador = torch.Generator().manual_seed(semilla)
    mascara_retenida = torch.zeros_like(mascara_valido)
    n_canales = mascara_valido.shape[-1]
    for c in range(n_canales):
        idx_validos = mascara_valido[:, :, c].nonzero(as_tuple=False)
        n_retener = int(idx_validos.shape[0] * fraccion)
        perm = torch.randperm(idx_validos.shape[0], generator=generador)[:n_retener]
        seleccionados = idx_validos[perm]
        mascara_retenida[seleccionados[:, 0], seleccionados[:, 1], c] = True
    mascara_entrenamiento = mascara_valido & (~mascara_retenida)
    return mascara_entrenamiento, mascara_retenida


def construir_grafo_pacientes(x_norm, k_vecinos):
    # Grafo de pacientes INDEPENDIENTE del GAT+GCN (ver cabecera de
    # 23_modelo_vgae.py): similitud coseno sobre el propio vector de
    # atributos ya imputado/normalizado (aplanado por paciente), no
    # sobre embeddings de un GAT entrenado. Reutiliza la FUNCION de
    # construccion k-NN de 21_modelo_gat_gcn.py (misma logica:
    # similitud coseno, top-k, simetrizado), no el grafo en si.
    n_pacientes = x_norm.shape[0]
    perfil_paciente = x_norm.reshape(n_pacientes, -1)
    return construir_grafo_knn_pacientes(perfil_paciente, k_vecinos)


def calcular_rmse_por_canal(x_reconstruido, x_verdad, mascara_retenida, canales):
    resultados = {}
    for c, nombre in enumerate(canales):
        m = mascara_retenida[:, :, c]
        if m.sum().item() == 0:
            resultados[nombre] = float("nan")
            continue
        diff2 = (x_reconstruido[:, :, c] - x_verdad[:, :, c]) ** 2
        resultados[nombre] = diff2[m].mean().sqrt().item()
    return resultados


def main(cohorte, ruta_salida, n_epocas, fraccion_retenida, k_vecinos, semilla,
         epocas_calentamiento_kl=0, minimo_libre_nats=0.5):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cohorte: {cohorte}")
    print("Cargando mascara de validez (datos crudos, ver 18_construir_atributos_gen.py)...")
    datos_crudos = cargar_atributos_crudos(cohorte)
    mascara_valido = datos_crudos["mascara_valido"]

    print("Cargando atributos ya imputados/normalizados (ver 20_preprocesar_atributos_gen.py)...")
    datos_norm = cargar_atributos_normalizados(cohorte)
    x_norm = datos_norm["x"]
    canales = datos_norm["canales"]
    n_pacientes, n_genes, n_canales = x_norm.shape
    print(f"  {n_pacientes} pacientes, {n_genes} genes, {n_canales} canales")

    print(f"Construyendo mascara sintetica de validacion "
          f"({fraccion_retenida * 100:.0f}% de las celdas validas, estratificada por canal)...")
    mascara_entrenamiento, mascara_retenida = construir_mascara_sintetica(mascara_valido, fraccion_retenida, semilla)
    for c, nombre in enumerate(canales):
        print(f"  {nombre}: {mascara_entrenamiento[:, :, c].sum().item()} entrenamiento, "
              f"{mascara_retenida[:, :, c].sum().item()} retenidas para validacion")

    print(f"Construyendo grafo de pacientes (k-NN independiente del GAT, k={k_vecinos})...")
    edge_index_pacientes = construir_grafo_pacientes(x_norm, k_vecinos)
    print(f"  {n_pacientes} pacientes, {edge_index_pacientes.shape[1]} aristas dirigidas")

    x_valores_plano = x_norm.reshape(n_pacientes, -1)
    mascara_entrenamiento_plano = mascara_entrenamiento.reshape(n_pacientes, -1)
    x_entrada_valores = torch.where(
        mascara_entrenamiento_plano, x_valores_plano, torch.zeros_like(x_valores_plano)
    )
    x_mascara_plano = mascara_entrenamiento_plano.float()

    modelo = ModeloVGAE(n_genes, n_canales)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    # KL annealing (calentamiento): beta_kl crece linealmente de 0 a 1
    # durante las primeras 'epocas_calentamiento_kl' epocas. Por
    # defecto (parametro en 0), se calcula como el 20% de n_epocas (10
    # de 50 en el pre-entrenamiento completo) - PERO ese calculo
    # automatico escala proporcionalmente con n_epocas, lo que en un
    # smoke test corto (ej. 5 epocas) da epocas_calentamiento_kl=1: un
    # UNICO salto de beta_kl=0 a 1 entre la epoca 1 y la 2, no una
    # rampa gradual real. Visto en la practica (primer smoke test con
    # annealing, ver runlog del Bloque 7): ese salto reproduce
    # practicamente el mismo colapso que sin annealing, simplemente
    # retrasado una epoca - NO es una prueba justa del esquema de 10
    # epocas de rampa que se usara en el entrenamiento real. Por eso
    # este parametro permite FIJAR epocas_calentamiento_kl de forma
    # explicita (ej. a 10) independientemente de n_epocas, para poder
    # hacer un smoke test con la MISMA granularidad de rampa que la
    # ejecucion real aunque el smoke test tenga menos epocas totales.
    #
    # Sin calentamiento alguno (beta_kl=1 desde la epoca 1): el
    # termino KL satura el clamp de logvar y domina la perdida por 4-5
    # ordenes de magnitud antes de que el encoder aprenda nada util de
    # la reconstruccion ("KL collapse" temprano, fenomeno conocido en
    # la literatura de VAEs, no un error de implementacion).
    if not epocas_calentamiento_kl:
        epocas_calentamiento_kl = max(1, round(n_epocas * 0.2))
    print(f"  free bits: minimo {minimo_libre_nats} nats de KL por dimension latente "
          f"(evita el posterior collapse, ver runlog del Bloque 7)")
    print(f"Pre-entrenamiento independiente, {n_epocas} epocas (Kipf y Welling 2016), "
          f"calentamiento KL en las primeras {epocas_calentamiento_kl} epocas...")
    for epoca in range(n_epocas):
        modelo.train()
        optimizador.zero_grad()
        beta_kl = min(1.0, epoca / epocas_calentamiento_kl)
        x_reconstruido, mu, logvar = modelo(x_entrada_valores, x_mascara_plano, edge_index_pacientes)
        perdida, perdida_recon, kl = perdida_elbo(
            x_reconstruido, x_valores_plano, mascara_entrenamiento_plano, mu, logvar,
            beta_kl=beta_kl, minimo_libre_nats=minimo_libre_nats,
        )
        perdida.backward()
        # Clip de gradiente como segunda red de seguridad numerica,
        # ademas del clamp de logvar en 23_modelo_vgae.py (visto en la
        # practica que el ELBO puede divergir a NaN sin el): limite
        # estandar en entrenamiento de GNN/VAE, no ajustado especificamente.
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
        optimizador.step()
        print(f"  epoca {epoca + 1}/{n_epocas}: beta_kl={beta_kl:.3f} elbo={perdida.item():.4f} "
              f"(reconstruccion={perdida_recon.item():.4f}, kl={kl.item():.4f})")

    print("Evaluando calidad de imputacion sobre la mascara sintetica retenida...")
    modelo.eval()
    with torch.no_grad():
        x_reconstruido, _mu, _logvar = modelo(x_entrada_valores, x_mascara_plano, edge_index_pacientes)
    x_reconstruido_3d = x_reconstruido.reshape(n_pacientes, n_genes, n_canales)
    rmse_por_canal = calcular_rmse_por_canal(x_reconstruido_3d, x_norm, mascara_retenida, canales)

    print("Generando tensor final imputado (valor real donde habia dato, reconstruccion del VGAE donde no)...")
    with torch.no_grad():
        # Reconstruccion final CON toda la mascara real visible (no la
        # de entrenamiento, que ocultaba el 10-15% retenido para
        # validacion): una vez evaluada arriba la calidad de
        # imputacion, el tensor que se guarda para el resto del
        # pipeline debe aprovechar TODO el dato real disponible.
        mascara_valido_plano = mascara_valido.reshape(n_pacientes, -1)
        x_entrada_final = torch.where(
            mascara_valido_plano, x_valores_plano, torch.zeros_like(x_valores_plano)
        )
        mascara_final_plano = mascara_valido_plano.float()
        x_reconstruido_final, _mu, _logvar = modelo(x_entrada_final, mascara_final_plano, edge_index_pacientes)
    x_reconstruido_final_3d = x_reconstruido_final.reshape(n_pacientes, n_genes, n_canales)
    x_imputado = torch.where(mascara_valido, x_norm, x_reconstruido_final_3d)

    ruta_tensor = Path("data/processed") / f"{cohorte}_atributos_gen_vgae.pt"
    torch.save({
        "cohorte": cohorte,
        "pacientes": datos_norm["pacientes"],
        "genes_ensg": datos_norm["genes_ensg"],
        "canales": canales,
        "x": x_imputado,
        "origen": "VGAE (Bloque 7): valor real donde mascara_valido=True, reconstruccion del VGAE en el resto",
    }, ruta_tensor)

    lineas = [
        f"Cohorte: {cohorte}",
        f"Pacientes: {n_pacientes}, genes: {n_genes}, canales: {n_canales}",
        f"Epocas de pre-entrenamiento: {n_epocas}, k vecinos (grafo pacientes): {k_vecinos}, "
        f"fraccion retenida para validacion: {fraccion_retenida}",
        "",
        f"ELBO final: {perdida.item():.4f} (reconstruccion={perdida_recon.item():.4f}, kl={kl.item():.4f})",
        "",
        "RMSE de imputacion por canal (sobre la mascara sintetica retenida, valor real vs reconstruido):",
    ]
    for nombre, rmse in rmse_por_canal.items():
        lineas.append(f"  {nombre}: {rmse:.4f}")
    lineas.append("")
    lineas.append(f"Tensor imputado guardado en: {ruta_tensor}")

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
    fraccion_retenida = float(argumentos[3]) if len(argumentos) > 3 else 0.12
    k_vecinos = int(argumentos[4]) if len(argumentos) > 4 else 20
    semilla = int(argumentos[5]) if len(argumentos) > 5 else 0
    # epocas_calentamiento_kl: 0 = automatico (20% de n_epocas); fijar
    # explicitamente (ej. 10) para que un smoke test corto use la
    # MISMA granularidad de rampa que una ejecucion mas larga (ver
    # comentario junto al calculo de epocas_calentamiento_kl en main()).
    epocas_calentamiento_kl = int(argumentos[6]) if len(argumentos) > 6 else 0
    # minimo_libre_nats: suelo de "free bits" por dimension latente
    # (Kingma et al. 2016), 0.5 nats por defecto (valor estandar de la
    # literatura), para evitar el posterior collapse (ver runlog del
    # Bloque 7). 0.0 desactiva free bits (comportamiento anterior).
    minimo_libre_nats = float(argumentos[7]) if len(argumentos) > 7 else 0.5
    sys.exit(main(cohorte, ruta_salida, n_epocas, fraccion_retenida, k_vecinos, semilla,
                  epocas_calentamiento_kl, minimo_libre_nats))
