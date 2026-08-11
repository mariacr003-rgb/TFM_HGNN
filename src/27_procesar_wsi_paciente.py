import sys, time, importlib, resource
from pathlib import Path
import numpy as np
import torch
import openslide

# Bloque 8: procesa UNA WSI de un paciente (teselado a parches de
# 256x256, filtrado de tejido, ResNet-50, pooling de atencion MIL) y
# guarda z_WSI + tiempos de cada fase por separado, para poder evaluar
# el coste real de extremo a extremo ANTES de comprometerse al volumen
# completo de las 5 cohortes (ver plan de diseño del Bloque 8).
#
# Filtrado de tejido: heuristica simple y estandar en la literatura de
# patologia computacional (ej. CLAM, Lu et al. 2021) - un parche se
# descarta si la mayoria de sus pixeles son "casi blancos" (fondo de
# cristal del portaobjetos, sin tejido), usando el canal de saturacion
# en HSV como proxy barato (el tejido tenido con H&E tiene saturacion
# alta; el fondo blanco tiene saturacion casi nula).
#
# PROCESAMIENTO EN STREAMING (corregido tras el primer intento real
# sobre TCGA-A1-A0SB, ver runlog paso44): la version original
# acumulaba TODOS los parches crudos (arrays 256x256x3 uint8, ~192KB
# cada uno) en una lista antes de convertirlos a tensor y pasarlos por
# ResNet-50, lo que agoto la RAM de la maquina (5.7GB) antes de
# terminar el teselado de un solo paciente (parche crudo x ~30-40k
# parches con tejido = varios GB), sin llegar siquiera a arrancar
# ResNet-50. Corregido calculando el embedding h_k de cada lote de
# parches (TAMANO_LOTE) en cuanto se completa, y descartando
# inmediatamente los arrays de imagen de ese lote: en memoria solo se
# acumulan los embeddings ya reducidos (2048 floats/parche, ~8KB) en
# vez de los parches crudos (~192KB/parche, 24x mas).

sys.path.insert(0, str(Path(__file__).resolve().parent))
_mod26 = importlib.import_module("26_modelo_mil")
EncoderResNetMIL = _mod26.EncoderResNetMIL
AtencionMIL = _mod26.AtencionMIL

TAMANO_PARCHE = 256
TAMANO_LOTE = 16
UMBRAL_SATURACION = 0.08  # fraccion minima de pixeles "con tejido" (saturacion > 0.1) para conservar un parche
MEDIA_IMAGENET = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD_IMAGENET = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pico_memoria_mb():
    # ru_maxrss: pico de RSS del proceso desde que arranco, en KB en
    # Linux (monotono no decreciente) - sirve para reportar el pico
    # real de memoria alcanzado en cada punto de control, sin depender
    # de psutil (no es una dependencia del proyecto).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def es_parche_con_tejido(parche_rgb_uint8):
    # parche_rgb_uint8: array HxWx3 uint8. Conversion manual RGB->HSV
    # (solo el canal de saturacion) para no depender de opencv/PIL para esto.
    rgb = parche_rgb_uint8.astype(np.float32) / 255.0
    maximo = rgb.max(axis=-1)
    minimo = rgb.min(axis=-1)
    saturacion = np.where(maximo > 0, (maximo - minimo) / np.clip(maximo, 1e-6, None), 0.0)
    fraccion_tejido = (saturacion > 0.1).mean()
    return fraccion_tejido > UMBRAL_SATURACION


def parches_a_tensor(parches):
    # Normalizacion estandar de ImageNet (necesaria porque el encoder
    # usa pesos preentrenados en ImageNet, ver 26_modelo_mil.py).
    t = torch.from_numpy(np.stack(parches)).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEDIA_IMAGENET) / STD_IMAGENET


def procesar_wsi_streaming(ruta_wsi, encoder, nivel=0, tamano=TAMANO_PARCHE,
                            tamano_lote=TAMANO_LOTE, max_parches=None):
    # nivel=0: resolucion maxima de la piramide del SVS. Recorre en
    # rejilla; cada vez que se acumulan tamano_lote parches con
    # tejido, calcula sus embeddings con ResNet-50 y los arrays crudos
    # de ese lote se descartan (salen de scope) antes de seguir
    # tesela ndo. Devuelve la lista de tensores de embeddings (uno por
    # lote), las coordenadas de cada parche y contadores de descarte.
    slide = openslide.OpenSlide(str(ruta_wsi))
    ancho, alto = slide.level_dimensions[nivel]
    print(f"  Dimensiones WSI (nivel {nivel}): {ancho} x {alto} pixeles "
          f"({(ancho // tamano) * (alto // tamano)} parches en rejilla completa)")

    n_celdas_totales = (ancho // tamano) * (alto // tamano)
    lote_parches, lote_coords = [], []
    embeddings, coords = [], []
    n_descartados_fondo = 0
    n_procesadas = 0
    n_con_tejido = 0
    t_inicio = time.time()

    def procesar_lote_actual():
        nonlocal lote_parches, lote_coords
        if not lote_parches:
            return
        x_lote = parches_a_tensor(lote_parches)
        with torch.no_grad():
            embeddings.append(encoder(x_lote))
        coords.extend(lote_coords)
        lote_parches = []
        lote_coords = []

    # Progreso periodico: sin esto, un corte de la VM de WSL (ver
    # runlog, incidente real durante el paciente de prueba) deja sin
    # ninguna cifra de avance real, ni para monitorizar en vivo ni
    # para saber cuanto se habia recorrido antes del corte. Se
    # reporta tambien el pico de RAM en cada punto de control, para
    # verificar en vivo que el streaming mantiene el consumo acotado
    # (a diferencia del primer intento, que agoto la RAM).
    for y in range(0, alto - tamano + 1, tamano):
        for x in range(0, ancho - tamano + 1, tamano):
            region = slide.read_region((x, y), nivel, (tamano, tamano)).convert("RGB")
            array = np.array(region)
            if es_parche_con_tejido(array):
                lote_parches.append(array)
                lote_coords.append((x, y))
                n_con_tejido += 1
                if len(lote_parches) >= tamano_lote:
                    procesar_lote_actual()
            else:
                n_descartados_fondo += 1
            n_procesadas += 1
            if n_procesadas % 5000 == 0:
                transcurrido = time.time() - t_inicio
                ritmo = n_procesadas / transcurrido
                restante = (n_celdas_totales - n_procesadas) / ritmo
                print(f"    {n_procesadas}/{n_celdas_totales} celdas ({n_con_tejido} con tejido) - "
                      f"{ritmo:.1f} celdas/s, ETA: {restante/60:.1f} min, "
                      f"pico RAM: {pico_memoria_mb():.0f} MB")
            if max_parches and n_con_tejido >= max_parches:
                procesar_lote_actual()
                slide.close()
                return embeddings, coords, n_descartados_fondo, n_con_tejido
    procesar_lote_actual()  # ultimo lote parcial, si quedo alguno sin llegar a tamano_lote
    slide.close()
    return embeddings, coords, n_descartados_fondo, n_con_tejido


def main(ruta_wsi, ruta_salida, max_parches):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    tiempos = {}
    picos_ram_mb = {}

    print(f"Procesando WSI: {ruta_wsi}")

    print("Cargando ResNet-50 (ImageNet, ver NOTA en 26_modelo_mil.py sobre el pretraining)...")
    t0 = time.time()
    encoder = EncoderResNetMIL(congelar_pesos=True)
    encoder.eval()
    tiempos["carga_resnet50"] = time.time() - t0
    picos_ram_mb["tras_carga_resnet50"] = pico_memoria_mb()

    print(f"Teselado + ResNet-50 en streaming (lotes de {TAMANO_LOTE} parches)...")
    t0 = time.time()
    embeddings, coords, n_descartados, n_con_tejido = procesar_wsi_streaming(
        ruta_wsi, encoder, max_parches=max_parches)
    tiempos["teselado_y_resnet50_streaming"] = time.time() - t0
    picos_ram_mb["tras_teselado_y_resnet50"] = pico_memoria_mb()
    print(f"  Parches con tejido: {n_con_tejido}, descartados por fondo: {n_descartados}")
    print(f"  Tiempo teselado+ResNet-50 combinado: {tiempos['teselado_y_resnet50_streaming']:.1f}s")
    if n_con_tejido:
        print(f"  ({tiempos['teselado_y_resnet50_streaming'] / n_con_tejido * 1000:.0f} ms/parche con tejido, "
              f"incluye lectura+filtrado+inferencia)")

    if not embeddings:
        raise ValueError("No se encontro ningun parche con tejido suficiente.")

    h = torch.cat(embeddings, dim=0)  # [n_parches, 2048]

    print("Calculando pooling de atencion MIL (Ilse et al. 2018)...")
    t0 = time.time()
    atencion = AtencionMIL(dim_entrada=encoder.dim_salida)
    atencion.eval()
    with torch.no_grad():
        z_wsi, pesos_atencion = atencion(h)
    tiempos["atencion_mil"] = time.time() - t0
    picos_ram_mb["tras_atencion_mil"] = pico_memoria_mb()

    tiempo_total = sum(tiempos.values())
    torch.save({
        "ruta_wsi": str(ruta_wsi),
        "n_parches": h.shape[0],
        "n_parches_descartados_fondo": n_descartados,
        "coords_parches": coords,
        "z_wsi": z_wsi,
        "pesos_atencion": pesos_atencion,
        "tiempos_segundos": tiempos,
        "tiempo_total_segundos": tiempo_total,
        "pico_ram_mb": picos_ram_mb,
    }, ruta_salida)

    print()
    print("Tiempos por fase:")
    for fase, t in tiempos.items():
        print(f"  {fase}: {t:.1f}s")
    print(f"  TOTAL (sin descarga): {tiempo_total:.1f}s ({tiempo_total / 60:.1f} min)")
    print("Pico de RAM por punto de control:")
    for punto, mb in picos_ram_mb.items():
        print(f"  {punto}: {mb:.0f} MB")
    print(f"z_wsi: {tuple(z_wsi.shape)}")
    print(f"Guardado: {ruta_salida}")
    return 0


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    ruta_wsi = argumentos[0]
    ruta_salida = argumentos[1]
    max_parches = int(argumentos[2]) if len(argumentos) > 2 and argumentos[2] else None
    sys.exit(main(ruta_wsi, ruta_salida, max_parches))
