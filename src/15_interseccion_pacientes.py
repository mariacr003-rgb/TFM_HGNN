import sys
from collections import defaultdict

# Selecciona, para cada paciente (case_id), un unico fichero por modalidad
# entre los candidatos ya resueltos por 14_mapear_case_id.py: exige que el
# tipo de muestra sea "Primary Tumor" (para no mezclar, para un mismo
# paciente, tumor en una modalidad con tejido normal/metastasico en otra) y,
# si aun asi hay mas de un candidato Primary Tumor por caso, desempata
# primero por prioridad de pipeline (solo CNV: fichero plano > ascat3 >
# absolute_liftover) y despues por file_id (UUID) alfabeticamente menor.
# Calcula despues la interseccion de casos con las 3 modalidades disponibles.
#
# Para metilacion se exige ademas que el campo "platform" sea el array
# estandar del proyecto (Illumina 450K, ver Paso 11): esto evita repetir el
# bug de mezclar arrays 27K/450K/EPIC (paso desapercibido en la muestra
# inicial de BRCA/LUAD porque el filtro solo miraba el nombre de fichero).
# El filtro por "platform" (campo real del GDC, verificado en el Paso 31)
# sustituye al proxy heuristico de tamano de fichero en bytes usado hasta
# ahora para construir la lista de entrada de metilacion.

ARRAY_METILACION_ESTANDAR = "Illumina Human Methylation 450"


def leer(ruta):
    # Formato esperado: salida de 14_mapear_case_id.py (5 columnas
    # separadas por tab, sin cabecera): file_id, filename, case_id,
    # sample_type, platform
    filas = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 5 or not partes[2]:
                continue
            fid, filename, case_id, tipo_muestra, plataforma = partes
            filas.append((fid, filename, case_id, tipo_muestra, plataforma))
    return filas


def filtrar_array_metilacion(filas):
    descartados = [f for f in filas if f[4] != ARRAY_METILACION_ESTANDAR]
    validos = [f for f in filas if f[4] == ARRAY_METILACION_ESTANDAR]
    if descartados:
        plataformas_descartadas = sorted(set(f[4] for f in descartados))
        print(f"  descartados {len(descartados)} ficheros de metilacion por array distinto de '{ARRAY_METILACION_ESTANDAR}': {plataformas_descartadas}")
    return validos


def prioridad_cnv(filename):
    if "absolute_liftover" in filename:
        return 2
    if "ascat3" in filename:
        return 1
    return 0


def filtrar_tumor_primario(filas):
    return [f for f in filas if "Primary Tumor" in f[3].split(";")]


def elegir_unico_por_caso(filas, con_prioridad_cnv=False):
    candidatos = defaultdict(list)
    for fid, filename, case_id, tipo, plataforma in filas:
        p = prioridad_cnv(filename) if con_prioridad_cnv else 0
        candidatos[case_id].append((p, fid, filename))

    mejor = {}
    casos_con_empate = 0
    for case_id, opciones in candidatos.items():
        opciones.sort(key=lambda o: (o[0], o[1]))
        p, fid, filename = opciones[0]
        mejor[case_id] = (fid, filename)
        if len(opciones) > 1:
            casos_con_empate += 1
    return mejor, casos_con_empate


def main(ruta_rnaseq, ruta_cnv, ruta_meth, out_path, n_max=None):
    rnaseq = filtrar_tumor_primario(leer(ruta_rnaseq))
    cnv = filtrar_tumor_primario(leer(ruta_cnv))
    meth_candidatos = leer(ruta_meth)
    print(f"Metilacion: {len(meth_candidatos)} ficheros candidatos antes de filtrar por array")
    meth = filtrar_tumor_primario(filtrar_array_metilacion(meth_candidatos))

    mejor_rnaseq, empates_rnaseq = elegir_unico_por_caso(rnaseq)
    mejor_cnv, empates_cnv = elegir_unico_por_caso(cnv, con_prioridad_cnv=True)
    mejor_meth, empates_meth = elegir_unico_por_caso(meth)

    print(f"RNA-seq: {len(rnaseq)} ficheros Primary Tumor -> {len(mejor_rnaseq)} casos unicos, {empates_rnaseq} con desempate")
    print(f"CNV: {len(cnv)} ficheros Primary Tumor -> {len(mejor_cnv)} casos unicos, {empates_cnv} con desempate")
    print(f"Metilacion: {len(meth)} ficheros Primary Tumor -> {len(mejor_meth)} casos unicos, {empates_meth} con desempate")

    interseccion = set(mejor_rnaseq) & set(mejor_cnv) & set(mejor_meth)
    print(f"Interseccion (Primary Tumor consistente, 1 fichero/caso/modalidad): {len(interseccion)}")

    seleccion = sorted(interseccion)
    if n_max is not None:
        seleccion = seleccion[:n_max]
        print(f"Guardando los primeros {len(seleccion)} casos (n_max={n_max})")
    else:
        print(f"Guardando la lista completa de {len(seleccion)} casos (sin n_max)")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("case_id\tfile_id_rnaseq\tfile_id_cnv\tfile_id_meth\n")
        for c in seleccion:
            f.write(f"{c}\t{mejor_rnaseq[c][0]}\t{mejor_cnv[c][0]}\t{mejor_meth[c][0]}\n")
    print(f"Guardado: {out_path}")
    return 0


if __name__ == "__main__":
    ruta_rnaseq, ruta_cnv, ruta_meth, out_path = sys.argv[1:5]
    n_max = int(sys.argv[5]) if len(sys.argv) > 5 else None
    sys.exit(main(ruta_rnaseq, ruta_cnv, ruta_meth, out_path, n_max))
