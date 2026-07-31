# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Trabajo Fin de Máster en Bioinformática (UAX): framework de integración
multi-ómica con Redes Neuronales de Grafos Heterogéneas (HGNN) para
predicción de supervivencia oncológica, usando datos reales de TCGA.

**Lee `CONTEXTO_PROYECTO.md` al empezar cualquier tarea en este repo** —
es la fuente de verdad sobre el alcance, el estado de avance y lo que
falta por implementar, y se actualiza con más frecuencia que este
fichero.

## Alcance objetivo

El proyecto debe implementar las 5 técnicas descritas en la Sección 4
del TFM (documento Word ya redactado, no incluido en este repositorio)
sobre las 5 cohortes de TCGA:

1. GAT (Graph Attention Network) — atención sobre genes
2. GCN (Graph Convolutional Network) — propagación entre pacientes
3. VGAE (Variational Graph Autoencoder) — imputación generativa de datos faltantes
4. MIL (Multiple Instance Learning con atención) — integración de imágenes WSI
5. DEC (Deep Embedded Clustering) — descubrimiento de subtipos moleculares

Cohortes: BRCA, LUAD, LUSC, COAD, KIRC.

**Estado actual: las 5 cohortes tienen ya sus 4 capas de datos reales
completas** (ver `docs/manifest-datos.tsv` para el detalle fichero a
fichero, y `results/2026-07-25-paso20/runlog.txt` para el resumen de
cierre de este bloque):

- Clínico: cohorte completa por proyecto GDC, número de pacientes
  distinto por cohorte según la Tabla 2 del TFM (BRCA 1.098, LUAD 585,
  LUSC 504, COAD 461, KIRC 537)
- RNA-seq: muestra real de 19.962 genes x 20 pacientes por cohorte
  (BRCA, LUAD y LUSC ya ampliadas a 200, ver hallazgo de coordinación
  abajo)
- CNV: muestra real de 60.624 genes x 20 pacientes por cohorte
  (BRCA, LUAD y LUSC ya ampliadas a 200, ver hallazgo de coordinación
  abajo)
- Metilación: muestra real de 486.427 sitios CpG x 20 pacientes por
  cohorte, todas del mismo array Illumina 450K (ver hallazgo sobre el
  array abajo; BRCA, LUAD y LUSC ya ampliadas a 200, ver hallazgo de
  coordinación abajo)

IMPORTANTE — hallazgo y corrección sobre la coordinación de pacientes
entre modalidades (BRCA): los 20 pacientes de RNA-seq, CNV y
metilación de BRCA se habían seleccionado de forma independiente
(primeros N ficheros de cada modalidad que casan con su patrón de
nombre en el manifest completo), sin exigir que fueran el mismo
paciente en las 3 tablas — al cruzarlas, solo 1 paciente coincidía en
las 3 simultáneamente, lo que invalida cualquier análisis multi-ómico
por paciente. Corregido en el Paso 24: se identificaron 786 case ID de
BRCA con las 3 modalidades disponibles y se seleccionaron los primeros
200 en orden alfabético (con prioridad plana > ascat3 >
absolute_liftover para desambiguar CNV cuando había más de un fichero
por caso), ampliando también el tamaño de muestra de 20 a 200. Esto
además forzó una reescritura de `src/11_convertir_metilacion.py` a
procesamiento en streaming (200 ficheros abiertos a la vez con
`zip_longest`, fila a fila, sin cargar las tablas completas en
memoria), porque la versión anterior moría por falta de RAM
("Killed") al intentar cargar 486.427 sitios CpG x 200 pacientes de
golpe. Ver `results/2026-07-31-paso24/runlog.txt` para el detalle
completo. Replicado en LUAD en el Paso 25 (`results/2026-07-31-paso25/runlog.txt`),
con un matiz adicional descubierto en ese paso: el manifest de LUAD no
incluye el case ID del paciente en el nombre de fichero de RNA-seq ni
metilación (a diferencia de lo asumido para BRCA), por lo que hubo que
resolver el case ID de cada fichero consultando la API del GDC; además
RNA-seq y metilación también tenían varios ficheros candidatos por
paciente (no solo CNV), resuelto filtrando a muestra "Primary Tumor" y,
si aun así quedaba más de un fichero por caso, con un desempate
determinista por file_id. Replicado también en LUSC en el Paso 26
(`results/2026-08-01-paso26/runlog.txt`), aplicando ya desde el
principio el criterio completo (consulta a la API del GDC, filtro
"Primary Tumor" y desempate por file_id en las 3 modalidades) sin
necesidad de descubrirlo sobre la marcha. Pendiente: replicar esta
misma coordinación de pacientes entre modalidades en COAD y KIRC
(siguen con 20 pacientes por modalidad, sin verificar si están
coordinados entre sí).

IMPORTANTE — hallazgo y corrección sobre la metilación: la primera
descarga de metilación de BRCA (Paso 5) y el primer intento de LUAD
mezclaban, sin que nadie se diera cuenta, 3 generaciones distintas del
array de metilación de Illumina (27K/450K/EPIC) — el filtro solo
miraba el nombre de fichero, y `src/11_convertir_metilacion.py` no
compara el identificador de sonda (`cg_id`) entre pacientes, por lo
que el error no producía ningún fallo visible, solo una tabla
combinada científicamente invalida. Se detectó y corrigió en el
Paso 11: se fijó el array 450K como estándar para las 5 cohortes,
filtrando por tamaño de fichero (rango 12.000.000-14.000.000 bytes
sobre el campo `size` del manifest del GDC, que coincide exacto con
el tamaño real descargado) antes de lanzar la descarga. Verificado sin
incidencias, aplicando el filtro desde el principio, en LUSC/COAD/KIRC
(Pasos 14, 17, 20). Ver `results/2026-07-24-paso11/runlog.txt` para el
analisis completo, incluida la transparencia sobre el origen de los
valores de referencia de los arrays 27K/EPIC.

Trabajo pendiente: MIL, DEC y el entrenamiento de GAT/GCN/VGAE sobre
las 5 cohortes (de momento solo probado minimamente en BRCA con el
script 12). STRING v12 ya está descargado; Reactome y KEGG siguen
pendientes para completar el grafo heterogéneo de vías (Fase 3).
`docs/manifest-datos.tsv` ya está al día con este estado; `README.md`
NO — todavía dice "solo está descargada la cohorte TCGA-BRCA" y no
refleja las 5 cohortes completas, pendiente de actualizar.

## Comandos

No hay build/lint/test automatizado (ni Makefile, ni CI, ni framework
de tests): la verificación de cada script es manual y se registra en
`results/<fecha>-pasoN/runlog.txt`.

Dos entornos Python separados según la fase:

```
# Fase 1 - pipeline de datos (Python del sistema, Windows, 3.14)
python src/00_generar_datos_prueba.py docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv
python src/01_validate_data.py PRUEBA docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv data/processed
python src/01_validate_data.py BRCA data/raw/BRCA_clinical.tsv data/raw/BRCA_rnaseq.tsv data/processed

# Fase 2 - framework GAT+GCN+VGAE (entorno virtual venv_pytorch, Python 3.11)
venv_pytorch\Scripts\activate
python src/12_prueba_pytorch_geometric.py
```

Dependencias en `entorno/requirements.txt`: `matplotlib` para Fase 1;
`torch==2.13.0` + `torch_geometric==2.8.0` (CPU, sin GPU disponible)
para Fase 2.

IMPORTANTE: al validar datos de prueba con `01_validate_data.py`, usar
siempre la etiqueta de cohorte `PRUEBA` (nunca `BRCA` ni otro código de
cohorte real) para no mezclar salidas ficticias con `data/processed/`.

## Arquitectura y flujo de datos

Pipeline lineal de scripts numerados en `src/` (`00_`, `01_`, `02_`...),
cada uno ejecutable de forma independiente por CLI con argumentos
posicionales (`sys.argv`), sin librería de parsing de argumentos:

- `00_generar_datos_prueba.py` → genera datos ficticios (`docs/tests/`)
- `01_validate_data.py` → valida clínicos + una capa ómica (filtra
  pacientes sin seguimiento, filtra genes con >50% valores vacíos)
- `02_filtrar_manifest.py` / `02b` (CNV) / `02c` (metilación) → filtran
  el manifest completo del GDC (no versionado) por patrón de nombre de
  fichero, para elegir qué descargar con `gdc-client.exe`. `02c` acepta
  además dos argumentos posicionales opcionales, `size_min` y
  `size_max` (bytes), para filtrar también por tamaño de fichero y así
  elegir un array de metilación concreto — p. ej. `... 20 12000000
  14000000` para quedarse solo con el array 450K; sin esos argumentos
  se comporta igual que antes (uso normal en 02 y 02b: sin size_min/max)
- `03_convertir_rnaseq.py`, `10_convertir_cnv.py`,
  `11_convertir_metilacion.py` → convierten los ficheros crudos
  descargados del GDC (uno por paciente, en subcarpetas con UUID) a
  una tabla única por cohorte en `data/raw/`
- `04_convertir_clinical.py` → convierte el fichero clínico completo
  del GDC a las 6 columnas del pipeline; detecta automáticamente el
  formato de entrada (`.tsv` plano, `.tsv.gz` gzip simple, o un
  paquete `tar.gz` con varias tablas — usa solo el miembro
  `clinical.tsv` del tar, sin necesidad de descomprimir a mano)
- `05_estadisticas_descriptivas.py`, `06_grafico_supervivencia.py` →
  estadísticas y figuras reales sobre los datos validados
- `07_figura4_cindex.py`, `08_figura6_subtipos_paad.py`,
  `09_figura7_roc.py` → generan figuras de resultados, pero **todavía
  con datos de ejemplo ilustrativos, sin adaptar a datos reales**
- `12_prueba_pytorch_geometric.py` → primer script de Fase 2 (PyG)

IMPORTANTE — convención de nombres de los manifests completos en
`data/raw/`: las 5 cohortes siguen el mismo patrón
`<cohorte>_manifest_full.txt` en minúsculas: `brca_manifest_full.txt`,
`luad_manifest_full.txt`, `lusc_manifest_full.txt`,
`coad_manifest_full.txt`, `kirc_manifest_full.txt`. El de BRCA se
llamó originalmente `gdc_manifest_full.txt` (sin prefijo de cohorte,
por ser el primero descargado, antes de que existieran las otras 4
cohortes) y se renombró despues a `brca_manifest_full.txt` para
seguir el mismo patrón; todas las referencias en runlogs, README.md y
`docs/manifest-datos.tsv` se actualizaron al renombrar, asi que el
nombre `gdc_manifest_full.txt` ya no deberia aparecer en ningun sitio
del repositorio.

Flujo de datos: `data/raw/` (descargas GDC/STRING, muchas no
versionadas por tamaño — ver `.gitignore`) → `data/processed/` (salida
de validación, por cohorte: `<COHORTE>_clinical_valid.tsv`,
`<COHORTE>_rnaseq_valid.tsv`, `<COHORTE>_validation_report.txt`) →
`results/<fecha>-pasoN/` (figuras, tablas, runlog).

`docs/manifest-datos.tsv` documenta origen, rol y URL de cada fichero
de datos: cubre ya las 4 capas completas de las 5 cohortes (clínico,
RNA-seq, CNV, metilación) más STRING; Reactome, KEGG y WSI siguen
pendientes de añadir cuando se incorporen (Fase 3).

Idempotencia: cada script relevante registra el hash SHA-256 de sus
ficheros de entrada en `logs/manifest.json` para detectar si ya se
ejecutó con los mismos datos y omitir el recálculo.

## Convención de trazabilidad (runlogs)

Cada paso de trabajo relevante se documenta en
`results/<fecha>-pasoN/runlog.txt` con: ID de ejecución, fecha,
objetivo, comando(s) exacto(s) ejecutado(s), fichero(s) de salida,
commit de Git correspondiente y observaciones/conclusión. Ver
`results/2026-06-20-paso5/runlog.txt` como ejemplo de formato. Al
completar un paso de trabajo nuevo, crear un runlog siguiendo este
mismo formato en vez de solo dejar el resultado en el código.

## Estilo de código

- Scripts simples y lineales, sin clases, funciones básicas
- `csv` estándar de Python (no pandas) para el pipeline de datos de Fase 1
- Cada script numerado secuencialmente en `src/`, ejecutable de forma independiente
- Comentarios y nombres de variable en español, sin tildes en el código
