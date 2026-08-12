import sys, importlib, time
from pathlib import Path
import torch

# Bloque 9 (DEC): replica el barrido de 90 configuraciones ya hecho a
# mano sobre BRCA en results/2026-08-11-paso47/runlog.txt (K en
# {2,3,4,5,6} x learning_rate en {0,001; 0,01; 0,05} x intervalo de
# actualizacion de P en {5, 20} x 3 semillas), generalizado a
# cualquier cohorte, para poder aplicarlo tambien a LUAD/LUSC/COAD/KIRC.
#
# Reutiliza ejecutar_dec() de 28_entrenar_dec.py (via importlib, mismo
# patron que 25 usa para importar 21/22) en vez de duplicar el bucle
# de refinamiento DEC.
#
# Criterio de decision (endurecido en el paso 49, ver
# results/2026-08-12-paso49/runlog.txt): se considera "no degenerado"
# cualquier configuracion con entropia normalizada >= 0,7, pero un
# K>=3 solo se acepta como estructura ROBUSTA si al menos la MITAD de
# sus 18 configuraciones (lr x intervalo_p x semilla) son no
# degeneradas - no basta con una sola. El criterio original (bastaba
# 1 de 18) resulto ser estadisticamente fragil: en LUAD, LUSC y COAD
# eligio un K>=3 sostenido por solo 1-4 de 18 configuraciones (ruido
# de comparaciones multiples: probar 18 combinaciones y quedarse con
# el maximo casi garantiza que alguna cruce un umbral fijo por azar),
# mientras que K=2 es no degenerado en el 100% de sus configuraciones
# en las 5 cohortes probadas hasta ahora (BRCA, LUAD, LUSC, COAD,
# KIRC). Si ningun K>=3 alcanza la mayoria, se entrega la mejor
# configuracion de K=2, documentando el mismo colapso que en BRCA.

sys.path.insert(0, str(Path(__file__).resolve().parent))
_mod28 = importlib.import_module("28_entrenar_dec")
ejecutar_dec = _mod28.ejecutar_dec
N_ITER = _mod28.N_ITER
UMBRAL_CONVERGENCIA = _mod28.UMBRAL_CONVERGENCIA

KS = [2, 3, 4, 5, 6]
LRS = [0.001, 0.01, 0.05]
INTERVALOS_P = [5, 20]
SEMILLAS = [0, 1, 2]
UMBRAL_ENTROPIA_NO_DEGENERADO = 0.7  # mismo umbral que paso47
UMBRAL_MAYORIA_ROBUSTA = 0.5  # fraccion minima de configs de un K que deben ser no degeneradas (ver runlog paso49)


def main(cohorte, ruta_modelo_gat_gcn, ruta_salida):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cargando embedding h_final de {ruta_modelo_gat_gcn} (reutilizado de GAT+GCN, sin recalcular)...")
    d = torch.load(ruta_modelo_gat_gcn, map_location="cpu", weights_only=False)
    z = d["h_final"].detach().clone()
    pacientes = d["pacientes"]
    print(f"  z: {tuple(z.shape)} ({len(pacientes)} pacientes)")

    n_configs = len(KS) * len(LRS) * len(INTERVALOS_P) * len(SEMILLAS)
    print(f"Barrido de {n_configs} configuraciones (K x lr x intervalo_p x semilla), "
          f"mismo diseno que results/2026-08-11-paso47/runlog.txt (BRCA)...")

    t0 = time.time()
    resultados = []
    for k in KS:
        for lr in LRS:
            for intervalo_p in INTERVALOS_P:
                for semilla in SEMILLAS:
                    r = ejecutar_dec(z, k, lr, intervalo_p, N_ITER, UMBRAL_CONVERGENCIA, semilla)
                    resultados.append({
                        "k": k, "lr": lr, "intervalo_p": intervalo_p, "semilla": semilla,
                        "entropia": r["entropia_normalizada"],
                        "conteos": r["conteos_cluster"],
                        "iteracion_convergencia": r["iteracion_convergencia"],
                        "asignacion": r["asignacion"],
                        "centros": r["centros"],
                    })
    duracion = time.time() - t0
    print(f"Barrido completo: {duracion:.1f}s ({n_configs} configuraciones)")

    print()
    print("Resumen por K (entropia normalizada, min-max sobre lr x intervalo_p x semilla):")
    for k in KS:
        ents = [r["entropia"] for r in resultados if r["k"] == k]
        n_no_degenerado = sum(1 for e in ents if e >= UMBRAL_ENTROPIA_NO_DEGENERADO)
        print(f"  K={k}: entropia {min(ents):.3f}-{max(ents):.3f} "
              f"({n_no_degenerado}/{len(ents)} configs >= {UMBRAL_ENTROPIA_NO_DEGENERADO})")

    candidatos_k_mayor_robustos = []
    for k in KS:
        if k < 3:
            continue
        configs_k = [r for r in resultados if r["k"] == k]
        no_degenerados_k = [r for r in configs_k if r["entropia"] >= UMBRAL_ENTROPIA_NO_DEGENERADO]
        if len(no_degenerados_k) / len(configs_k) >= UMBRAL_MAYORIA_ROBUSTA:
            candidatos_k_mayor_robustos.append(max(no_degenerados_k, key=lambda r: r["entropia"]))

    if candidatos_k_mayor_robustos:
        ganador = max(candidatos_k_mayor_robustos, key=lambda r: r["entropia"])
        n_k_configs = len(LRS) * len(INTERVALOS_P) * len(SEMILLAS)
        n_no_degenerados_ganador = sum(
            1 for r in resultados if r["k"] == ganador["k"] and r["entropia"] >= UMBRAL_ENTROPIA_NO_DEGENERADO
        )
        limitacion = (
            f"Barrido de {n_configs} configuraciones: a diferencia de BRCA (paso47), {cohorte} "
            f"SI tiene una mayoria robusta de configuraciones con K={ganador['k']} no degeneradas "
            f"({n_no_degenerados_ganador}/{n_k_configs} configs >= {UMBRAL_ENTROPIA_NO_DEGENERADO}, "
            f">= {UMBRAL_MAYORIA_ROBUSTA:.0%} requerido). Se entrega como resultado."
        )
    else:
        candidatos_k2 = [r for r in resultados if r["k"] == 2 and r["entropia"] >= UMBRAL_ENTROPIA_NO_DEGENERADO] or \
            [r for r in resultados if r["k"] == 2]
        ganador = max(candidatos_k2, key=lambda r: r["entropia"])
        limitacion = (
            f"Mismo patron que BRCA (results/2026-08-11-paso47/runlog.txt): de {n_configs} "
            f"configuraciones, ninguna con K>=3 alcanza entropia>={UMBRAL_ENTROPIA_NO_DEGENERADO} "
            f"(no degenerada) de forma consistente. K=2 (entropia={ganador['entropia']:.3f}) es el "
            f"resultado no degenerado entregado. No es el descubrimiento de multiples subtipos "
            f"moleculares buscado originalmente."
        )

    torch.save({
        "cohorte": cohorte,
        "pacientes": pacientes,
        "asignacion_cluster": ganador["asignacion"],
        "centros": ganador["centros"],
        "K": ganador["k"],
        "lr": ganador["lr"],
        "intervalo_actualizar_p": ganador["intervalo_p"],
        "semilla": ganador["semilla"],
        "entropia_normalizada": ganador["entropia"],
        "conteos_cluster": ganador["conteos"],
        "iteracion_convergencia": ganador["iteracion_convergencia"],
        "n_configuraciones_barridas": n_configs,
        "duracion_barrido_s": duracion,
        "LIMITACION": limitacion,
    }, ruta_salida)

    print()
    print(f"Configuracion ganadora: K={ganador['k']}, lr={ganador['lr']}, "
          f"intervalo_p={ganador['intervalo_p']}, semilla={ganador['semilla']}")
    print(f"Distribucion final: {ganador['conteos']}, entropia normalizada={ganador['entropia']:.3f}")
    print(f"Guardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    cohorte = argumentos[0]
    ruta_modelo_gat_gcn = argumentos[1]
    ruta_salida = argumentos[2]
    sys.exit(main(cohorte, ruta_modelo_gat_gcn, ruta_salida))
