import sys, math
from pathlib import Path
import torch

# Bloque 9: DEC (Deep Embedded Clustering, Xie, Girshick y Farhadi 2016,
# Seccion 4.6 del TFM) para descubrimiento de subtipos moleculares.
#
# Reutiliza DIRECTAMENTE el embedding "h_final" ya calculado y guardado
# por el prerrequisito del Bloque 8 (src/25_entrenar_gat_gcn_final.py,
# data/processed/<COHORTE>_modelo_gat_gcn_final.pt): no hace falta un
# autoencoder propio de pre-entrenamiento (la version clasica de DEC
# usa una SAE para esto) porque GAT+GCN ya produce un embedding de
# paciente de baja dimension (32) del que partir.
#
# Sin excepcion de estilo aqui (a diferencia de GAT/GCN/VGAE/MIL): el
# unico parametro entrenable de DEC son los centroides de cluster, un
# unico tensor - no hace falta una subclase de nn.Module, basta con un
# torch.nn.Parameter suelto pasado al optimizador.
#
# HALLAZGO Y DECISION (ver results/2026-08-11-paso47/runlog.txt para
# el detalle completo del barrido de 90 configuraciones): con el
# embedding de GAT+GCN de BRCA, cualquier K>=3 colapsa de forma
# reproducible (mismo split en las 78 configuraciones de K=3,4,5,6
# probadas, independiente de learning rate/intervalo/semilla) al MISMO
# reparto binario subyacente (~65-68 vs ~113-117 pacientes), dejando
# los clusters sobrantes vacios. La formula ya incluye la reponderacion
# 1/f_j de Xie et al. (ver calcular_p), diseñada precisamente para
# evitar esto, y aun asi colapsa - evidencia de que el embedding de
# GAT+GCN (optimizado para riesgo de Cox, Bloque 6) solo contiene una
# estructura de 2 grupos separable de forma robusta, no una estructura
# rica de multiples subtipos. Corregirlo con rigor (IDEC, Guo et al.
# 2017: perdida de reconstruccion adicional que regulariza contra el
# colapso) es un cambio de arquitectura fuera del presupuesto de
# tiempo disponible para este bloque (~0,5 dia de 9 totales, decision
# explicita del usuario) - NO se persigue. Se entrega K=2 (unico
# resultado no degenerado, estable en las 12 configuraciones probadas)
# como limitacion documentada, no como el descubrimiento de subtipos
# multiples que se buscaba originalmente.

K = 2  # unico valor no degenerado encontrado en el barrido (ver runlog paso47)
LR = 0.01
INTERVALO_ACTUALIZAR_P = 10
N_ITER = 300
UMBRAL_CONVERGENCIA = 0.001
SEMILLA = 0  # misma convencion del proyecto (22_entrenar_gat_gcn.py, 24_entrenar_vgae.py)


def kmeans_init(z, k, semilla):
    torch.manual_seed(semilla)
    n = z.shape[0]
    idx = torch.randperm(n)[:k]
    centros = z[idx].clone()
    for _ in range(50):
        dist = torch.cdist(z, centros)
        asignacion = dist.argmin(dim=1)
        for j in range(k):
            if (asignacion == j).sum() > 0:
                centros[j] = z[asignacion == j].mean(dim=0)
    return centros, asignacion


def calcular_q(z, mu):
    # Kernel t-Student (Van der Maaten y Hinton 2008, grados de libertad=1):
    # q_ij = (1+||z_i-mu_j||^2)^-1 / suma_j'(1+||z_i-mu_j'||^2)^-1
    dist2 = torch.cdist(z, mu) ** 2
    q = 1.0 / (1.0 + dist2)
    return q / q.sum(dim=1, keepdim=True)


def calcular_p(q):
    # Distribucion objetivo de Xie et al. 2016, con reponderacion por
    # frecuencia de cluster (1/f_j) para evitar que un cluster grande
    # domine la asignacion - ver HALLAZGO arriba: no evita el colapso
    # con este embedding.
    f = q.sum(dim=0)
    p = (q ** 2) / f
    return p / p.sum(dim=1, keepdim=True)


def entropia_normalizada(asignacion, k):
    conteos = torch.bincount(asignacion, minlength=k).float()
    props = conteos / conteos.sum()
    props_nz = props[props > 0]
    h = -(props_nz * props_nz.log()).sum().item()
    return h / math.log(k)


def main(cohorte, ruta_modelo_gat_gcn, ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cargando embedding h_final de {ruta_modelo_gat_gcn} (reutilizado de GAT+GCN, sin recalcular)...")
    d = torch.load(ruta_modelo_gat_gcn, map_location="cpu", weights_only=False)
    z = d["h_final"].detach().clone()
    pacientes = d["pacientes"]
    print(f"  z: {tuple(z.shape)} ({len(pacientes)} pacientes)")

    centros, asignacion_inicial = kmeans_init(z, K, SEMILLA)
    print(f"k-means init (K={K}): {torch.bincount(asignacion_inicial, minlength=K).tolist()}, "
          f"entropia normalizada={entropia_normalizada(asignacion_inicial, K):.3f}")

    mu = torch.nn.Parameter(centros.clone())
    optimizador = torch.optim.Adam([mu], lr=LR)
    torch.manual_seed(SEMILLA)

    with torch.no_grad():
        p = calcular_p(calcular_q(z, mu))
    asignacion_previa = asignacion_inicial
    iteracion_convergencia = None
    for it in range(N_ITER):
        if it % INTERVALO_ACTUALIZAR_P == 0 and it > 0:
            with torch.no_grad():
                p = calcular_p(calcular_q(z, mu))
        q = calcular_q(z, mu)
        perdida = torch.nn.functional.kl_div(q.log(), p, reduction="batchmean")
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        with torch.no_grad():
            asignacion_actual = calcular_q(z, mu).argmax(dim=1)
        frac_cambio = (asignacion_actual != asignacion_previa).float().mean().item()
        if frac_cambio < UMBRAL_CONVERGENCIA and iteracion_convergencia is None:
            iteracion_convergencia = it
        asignacion_previa = asignacion_actual

    ent_final = entropia_normalizada(asignacion_previa, K)
    conteos_final = torch.bincount(asignacion_previa, minlength=K).tolist()

    torch.save({
        "cohorte": cohorte,
        "pacientes": pacientes,
        "asignacion_cluster": asignacion_previa,
        "centros": mu.detach(),
        "K": K,
        "entropia_normalizada": ent_final,
        "conteos_cluster": conteos_final,
        "iteracion_convergencia": iteracion_convergencia,
        "LIMITACION": ("K=2 es el unico resultado no degenerado encontrado en el barrido de 90 "
                        "configuraciones (results/2026-08-11-paso47/runlog.txt); K>=3 colapsa de "
                        "forma reproducible al mismo reparto binario subyacente. No es el "
                        "descubrimiento de multiples subtipos moleculares buscado originalmente."),
    }, ruta_salida)

    print(f"Convergencia (<0,1% cambios de asignacion): iteracion {iteracion_convergencia}")
    print(f"Distribucion final: {conteos_final}, entropia normalizada={ent_final:.3f}")
    print(f"Guardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_modelo_gat_gcn = argumentos[1]
    ruta_salida = argumentos[2]
    sys.exit(main(cohorte, ruta_modelo_gat_gcn, ruta_salida))
