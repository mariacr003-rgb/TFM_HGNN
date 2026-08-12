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
# Criterio de decision (igual que en el barrido original de BRCA,
# paso47): se considera "no degenerado" cualquier configuracion con
# entropia normalizada >= 0,7. Si alguna configuracion con K>=3 lo
# alcanza, se entrega esa (senal de estructura de mas de 2 grupos,
# distinto del patron de BRCA); si no, se entrega la mejor
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

    candidatos_no_degenerados = [r for r in resultados if r["entropia"] >= UMBRAL_ENTROPIA_NO_DEGENERADO]
    candidatos_k_mayor = [r for r in candidatos_no_degenerados if r["k"] >= 3]

    if candidatos_k_mayor:
        ganador = max(candidatos_k_mayor, key=lambda r: r["entropia"])
        limitacion = (
            f"Barrido de {n_configs} configuraciones: a diferencia de BRCA (paso47), {cohorte} "
            f"SI tiene al menos una configuracion con K={ganador['k']} no degenerada "
            f"(entropia={ganador['entropia']:.3f} >= {UMBRAL_ENTROPIA_NO_DEGENERADO}). Se entrega "
            f"como resultado, pero sin mas verificacion de robustez que las 3 semillas probadas "
            f"en este barrido."
        )
    else:
        candidatos_k2 = [r for r in candidatos_no_degenerados if r["k"] == 2] or \
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
