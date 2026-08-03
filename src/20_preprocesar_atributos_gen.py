import sys
from pathlib import Path
import torch

# Imputa los NaN y normaliza los 3 canales de <COHORTE>_atributos_gen.pt
# (RNA-seq, CNV, metilacion; ver 18_construir_atributos_gen.py) antes de
# usarlos como entrada del GAT. El vector de atributos sigue teniendo 3
# canales (formula 5 del TFM, sin el 4o canal de mutacion somatica): este
# script solo transforma los valores, no anade ni quita canales.
#
# Imputacion: media del gen a traves de los pacientes de la cohorte,
# calculada SOLO con los valores validos (mascara_valido de
# 18_construir_atributos_gen.py). Si un gen no tiene NINGUN valor valido
# en toda la cohorte (ej. los 13 genes mitocondriales sin CNV, o genes
# sin ninguna sonda de promotor en metilacion), se usa como respaldo la
# media global del canal (media de las medias por gen que si existen).
#
# Normalizacion por canal:
#   - RNA-seq (TPM): log1p, luego z-score por gen (a traves de los
#     pacientes de la cohorte)
#   - CNV (copy_number): z-score por gen directamente (ya es una escala
#     razonable, sin transformacion previa)
#   - Metilacion (beta, [0,1]): transformacion logit a M-value
#     (log2(beta/(1-beta)), estandar en literatura de metilacion, mas
#     adecuada que beta cruda para modelos lineales/GNN por ser mas
#     simetrica), luego z-score por gen. beta se recorta a
#     [EPS_LOGIT, 1-EPS_LOGIT] antes del logit para evitar +-infinito.
#
# El z-score se calcula DESPUES de imputar (practica habitual de
# imputacion simple por media): los valores imputados quedan en la media
# del gen, es decir, en 0 tras estandarizar.

EPS_LOGIT = 1e-4
EPS_STD = 1e-8


def imputar_media_por_gen(valores, mascara):
    # valores, mascara: [genes, pacientes]. mascara puede contener NaN
    # en las posiciones invalidas de "valores"; se limpian ANTES de
    # sumar (NaN * 0 = NaN, no 0, asi que no basta con multiplicar por
    # la mascara sin este paso).
    mascara_bool = mascara.bool()
    valores_limpios = torch.where(mascara_bool, valores, torch.zeros_like(valores))

    conteo_validos = mascara_bool.sum(dim=1)  # [genes]
    suma_validos = valores_limpios.sum(dim=1)  # [genes]
    media_gen = suma_validos / conteo_validos.clamp(min=1)

    genes_sin_dato = conteo_validos == 0
    n_genes_sin_dato = int(genes_sin_dato.sum().item())
    if n_genes_sin_dato > 0:
        genes_con_dato = ~genes_sin_dato
        media_global = media_gen[genes_con_dato].mean() if genes_con_dato.any() else torch.tensor(0.0)
        media_gen = torch.where(genes_sin_dato, media_global, media_gen)

    imputados = torch.where(mascara_bool, valores, media_gen.unsqueeze(1).expand_as(valores))
    return imputados, n_genes_sin_dato


def normalizar_zscore_por_gen(valores):
    media = valores.mean(dim=1, keepdim=True)
    std = valores.std(dim=1, keepdim=True)
    std_seguro = torch.where(std < EPS_STD, torch.ones_like(std), std)
    return (valores - media) / std_seguro


def procesar_canal_rnaseq(x, mascara, idx_canal):
    valores = x[:, :, idx_canal].transpose(0, 1)  # [genes, pacientes]
    mask = mascara[:, :, idx_canal].transpose(0, 1)
    log_valores = torch.log1p(valores.clamp(min=0))
    imputados, n_sin_dato = imputar_media_por_gen(log_valores, mask)
    return normalizar_zscore_por_gen(imputados), n_sin_dato


def procesar_canal_cnv(x, mascara, idx_canal):
    valores = x[:, :, idx_canal].transpose(0, 1)
    mask = mascara[:, :, idx_canal].transpose(0, 1)
    imputados, n_sin_dato = imputar_media_por_gen(valores, mask)
    return normalizar_zscore_por_gen(imputados), n_sin_dato


def procesar_canal_metilacion(x, mascara, idx_canal):
    valores = x[:, :, idx_canal].transpose(0, 1)
    mask = mascara[:, :, idx_canal].transpose(0, 1)
    beta_recortado = valores.clamp(min=EPS_LOGIT, max=1 - EPS_LOGIT)
    m_valores = torch.log2(beta_recortado / (1 - beta_recortado))
    imputados, n_sin_dato = imputar_media_por_gen(m_valores, mask)
    return normalizar_zscore_por_gen(imputados), n_sin_dato


def main(cohorte):
    ruta_entrada = Path("data/processed") / f"{cohorte}_atributos_gen.pt"
    ruta_salida = Path("data/processed") / f"{cohorte}_atributos_gen_norm.pt"

    print(f"Cohorte: {cohorte}")
    print(f"Cargando {ruta_entrada}...")
    datos = torch.load(ruta_entrada, weights_only=False)
    x = datos["x"]  # [pacientes, genes, 3]
    mascara = datos["mascara_valido"]
    canales = datos["canales"]
    idx_rna = canales.index("rnaseq_tpm")
    idx_cnv = canales.index("cnv_copy_number")
    idx_met = canales.index("metilacion_promotor_beta")

    print("Canal 1/3: RNA-seq (log1p + z-score por gen)...")
    rna_norm, genes_sin_rna = procesar_canal_rnaseq(x, mascara, idx_rna)
    print(f"  genes sin ningun valor valido: {genes_sin_rna}")

    print("Canal 2/3: CNV (imputacion media por gen + z-score)...")
    cnv_norm, genes_sin_cnv = procesar_canal_cnv(x, mascara, idx_cnv)
    print(f"  genes sin ningun valor valido: {genes_sin_cnv}")

    print("Canal 3/3: metilacion (logit a M-value + imputacion media por gen + z-score)...")
    met_norm, genes_sin_met = procesar_canal_metilacion(x, mascara, idx_met)
    print(f"  genes sin ningun valor valido: {genes_sin_met}")

    x_norm = torch.stack([rna_norm, cnv_norm, met_norm], dim=-1).transpose(0, 1)  # [pacientes, genes, 3]

    salida = {
        "cohorte": cohorte,
        "pacientes": datos["pacientes"],
        "genes_ensg": datos["genes_ensg"],
        "canales": canales,
        "x": x_norm,
        "transformaciones": {
            "rnaseq_tpm": "log1p + z-score por gen (imputacion: media del gen)",
            "cnv_copy_number": "z-score por gen (imputacion: media del gen)",
            "metilacion_promotor_beta": "logit a M-value + z-score por gen (imputacion: media del gen en escala M)",
        },
    }
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(salida, ruta_salida)

    print()
    print(f"Guardado: {ruta_salida}")
    print(f"Tensor x_norm: {tuple(x_norm.shape)}")
    return 0


if __name__ == "__main__":
    cohorte = sys.argv[1]
    sys.exit(main(cohorte))
