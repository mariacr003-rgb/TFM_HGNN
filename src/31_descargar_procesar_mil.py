import sys, csv, subprocess, time, shutil
from pathlib import Path

# Bloque 8 (MIL): descarga y procesa las WSI de los N pacientes
# seleccionados de UNA cohorte (ver src/30_seleccionar_pacientes_mil.py
# y data/processed/<COHORTE>_pacientes_mil.tsv), un paciente cada vez.
#
# DISENO "descargar -> procesar -> borrar" (no "descargar todo, luego
# procesar todo"): el disco donde vive el repo (C:\, montado en
# /mnt/c) solo tenia 33 GB libres en el momento de este paso, frente a
# los 44,89 GB del volumen total de las 5 cohortes - no cabia todo a
# la vez. Procesando un paciente cada vez y borrando su WSI cruda
# (data/raw/*.svs, nunca versionada, ver .gitignore) en cuanto se
# guarda su z_wsi, el pico de disco se queda en, como mucho, el
# tamano de UN paciente (~2,8 GB en el peor caso visto).
#
# gdc-client.exe es un binario de Windows (ejecutado via interop de
# WSL): necesita la ruta de destino en formato Windows (wslpath -w),
# no POSIX (/mnt/c/...) - una ruta POSIX produce un error de
# "Unable to write to download directory" porque el cliente concatena
# el nombre de fichero temporal con una barra invertida.
#
# Fallos (descarga o procesamiento) se registran y se SIGUE con el
# siguiente paciente - no se aborta el resto de la cohorte por un
# fallo aislado (mismo criterio que otros bloques: documentar, no
# detener).

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAIZ = Path(__file__).resolve().parent.parent
GDC_CLIENT = RAIZ / "tools" / "gdc-client.exe"


def wslpath_windows(ruta_posix):
    return subprocess.check_output(["wslpath", "-w", str(ruta_posix)], text=True).strip()


def descargar(file_id, dir_destino):
    dir_destino.mkdir(parents=True, exist_ok=True)
    dir_destino_win = wslpath_windows(dir_destino)
    resultado = subprocess.run(
        [str(GDC_CLIENT), "download", file_id, "-d", dir_destino_win],
        capture_output=True, text=True, timeout=3600,
    )
    return resultado.returncode == 0, resultado.stdout + resultado.stderr


def procesar(ruta_svs, ruta_salida):
    resultado = subprocess.run(
        [sys.executable, "-u", str(RAIZ / "src" / "27_procesar_wsi_paciente.py"),
         str(ruta_svs), str(ruta_salida)],
        capture_output=True, text=True, timeout=12 * 3600,
    )
    return resultado.returncode == 0, resultado.stdout + resultado.stderr


def main(cohorte):
    ruta_lista = RAIZ / "data" / "processed" / f"{cohorte}_pacientes_mil.tsv"
    dir_raw = RAIZ / "data" / "raw" / "wsi_mil" / cohorte
    dir_salida = RAIZ / "data" / "processed" / "mil_wsi" / cohorte
    dir_salida.mkdir(parents=True, exist_ok=True)

    with open(ruta_lista, newline="", encoding="utf-8") as f:
        pacientes = list(csv.DictReader(f, delimiter="\t"))

    print(f"Cohorte {cohorte}: {len(pacientes)} pacientes en la lista")

    exitos, fallos_descarga, fallos_procesamiento, ya_hechos = [], [], [], []
    tiempo_acumulado_s = 0.0

    for p in pacientes:
        case_id, file_id, file_name = p["case_id"], p["file_id"], p["file_name"]
        ruta_salida = dir_salida / f"{case_id}_mil.pt"

        if ruta_salida.exists():
            print(f"[{case_id}] YA PROCESADO (reutilizado, no se descarga ni reprocesa): {ruta_salida}")
            ya_hechos.append(case_id)
            continue

        print(f"[{case_id}] descargando {file_name} ({int(p['file_size']) / 1e9:.2f} GB)...")
        t0 = time.time()
        dir_paciente = dir_raw / file_id
        try:
            ok_descarga, salida_descarga = descargar(file_id, dir_raw)
        except Exception as e:
            ok_descarga, salida_descarga = False, str(e)

        if not ok_descarga or not (dir_paciente / file_name).exists():
            print(f"[{case_id}] FALLO DE DESCARGA:")
            print(salida_descarga[-2000:])
            fallos_descarga.append(case_id)
            shutil.rmtree(dir_paciente, ignore_errors=True)
            continue

        t_descarga = time.time() - t0
        print(f"[{case_id}] descarga OK en {t_descarga:.1f}s. Procesando (teselado+ResNet-50+atencion)...")

        ruta_svs = dir_paciente / file_name
        t0 = time.time()
        try:
            ok_proc, salida_proc = procesar(ruta_svs, ruta_salida)
        except Exception as e:
            ok_proc, salida_proc = False, str(e)
        t_proc = time.time() - t0

        if not ok_proc:
            print(f"[{case_id}] FALLO DE PROCESAMIENTO (tras {t_proc:.1f}s):")
            print(salida_proc[-3000:])
            fallos_procesamiento.append(case_id)
            shutil.rmtree(dir_paciente, ignore_errors=True)
            continue

        shutil.rmtree(dir_paciente, ignore_errors=True)
        t_total = t_descarga + t_proc
        tiempo_acumulado_s += t_total
        exitos.append(case_id)
        print(f"[{case_id}] OK - descarga {t_descarga:.1f}s + procesamiento {t_proc:.1f}s "
              f"= {t_total:.1f}s ({t_total / 60:.1f} min). Guardado: {ruta_salida}")

    print()
    print(f"=== RESUMEN COHORTE {cohorte} ===")
    print(f"  Completados en esta ejecucion: {len(exitos)}/{len(pacientes)} ({exitos})")
    print(f"  Ya procesados de antes (reutilizados): {len(ya_hechos)} ({ya_hechos})")
    print(f"  FALLOS DE DESCARGA: {len(fallos_descarga)} ({fallos_descarga})")
    print(f"  FALLOS DE PROCESAMIENTO: {len(fallos_procesamiento)} ({fallos_procesamiento})")
    print(f"  Tiempo real acumulado (descarga+procesamiento, esta ejecucion): "
          f"{tiempo_acumulado_s:.1f}s ({tiempo_acumulado_s / 60:.1f} min, {tiempo_acumulado_s / 3600:.2f} h)")

    total_fallos = len(fallos_descarga) + len(fallos_procesamiento)
    return 1 if total_fallos else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
