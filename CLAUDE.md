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
- RNA-seq: 19.962 genes x 200 pacientes coordinados, en las 5
  cohortes (ver hallazgo de coordinación abajo)
- CNV: 60.623 genes x 200 pacientes coordinados, en las 5 cohortes
  (ver hallazgo de coordinación abajo; cifra corregida en el Paso 32
  de 60.624 a 60.623 — la anterior incluía una fila fantasma por un
  fallo en `src/10_convertir_cnv.py` que no saltaba la cabecera del
  fichero crudo, ver `results/2026-08-01-paso32/runlog.txt`)
- Metilación: 486.427 sitios CpG (array Illumina 450K) x 200
  pacientes coordinados, en las 5 cohortes (ver hallazgo sobre el
  array y hallazgo de coordinación abajo)

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
(`results/2026-08-01-paso26/runlog.txt`), en COAD en el Paso 27
(`results/2026-08-01-paso27/runlog.txt`) y en KIRC en el Paso 28
(`results/2026-08-01-paso28/runlog.txt`), aplicando desde el
principio (a partir de LUSC) el criterio completo: consulta a la API
del GDC para resolver case_id y sample_type, filtro a muestra "Primary
Tumor" en las 3 modalidades (no solo CNV) y desempate final
determinista por file_id cuando aun así queda más de un fichero por
caso. Número de candidatos con las 3 modalidades disponibles y muestra
"Primary Tumor" consistente, por cohorte: BRCA 779, LUAD 454, LUSC
368, COAD 295, KIRC 316 — en las 5 cohortes se tomaron los primeros
200 en orden alfabético de case_id.

IMPORTANTE — hallazgo y corrección adicional en BRCA (Paso 29): al
cerrar el bloque de trabajo de las 5 cohortes, se verificó BRCA con el
mismo método riguroso usado en LUAD-KIRC (traducir cada columna a su
case_id y sample_type vía la API del GDC), algo que no se había hecho
porque BRCA (Paso 24) se corrigió ANTES de introducir el filtro
"Primary Tumor" (introducido después, en LUAD). Resultado: los 200
case_id sí coincidían en las 3 tablas, pero 10 de los 200 pacientes
(5%) tenían tipo de muestra inconsistente entre modalidades — 5 casos
con metilación de tejido normal en vez de tumoral, 3 casos con RNA-seq
Y metilación de tejido normal (solo el CNV tenía señal tumoral), 1
caso con RNA-seq de tejido normal, y 1 caso mezclando tumor
metastásico (RNA-seq) con tumor primario (metilación). Corregido
sustituyendo esos 10 pacientes por los siguientes 10 candidatos
válidos (Primary Tumor consistente) de los 779 candidatos rigurosos de
BRCA, manteniendo sin tocar a los otros 190 pacientes ya correctos.
Ver `results/2026-08-01-paso29/runlog.txt` para el detalle completo,
incluida la tabla de los 10 casos originales.

**Con esto, las 5 cohortes (BRCA, LUAD, LUSC, COAD, KIRC) tienen ya
RNA-seq, CNV y metilación coordinadas a 200 pacientes cada una,
compartiendo exactamente el mismo conjunto de 200 case_id dentro de
cada cohorte y con muestra tumoral primaria consistente en las 3
modalidades** (verificado explícitamente con el método riguroso en
las 5 cohortes, no solo asumido por construcción). Se cierra así el
bloque de trabajo iniciado en el Paso 24.

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

Bloque 6 (GAT+GCN) — CERRADO (2026-08-07): implementado en
`src/19_proyectar_grafo_gen_gen.py` (proyección del PPI de STRING a
grafo gen-gen, 19.962 nodos, 457.162 aristas dirigidas),
`src/20_preprocesar_atributos_gen.py` (imputación por media +
normalización de los 3 canales por gen), `src/21_modelo_gat_gcn.py`
(`ModeloGATGCN`: GAT de 2 capas/8 cabezas sobre el grafo gen-gen →
pooling medio por paciente → k-NN dinámico coseno → GCN de 2 capas →
riesgo escalar; única excepción documentada a la convención "sin
clases" del proyecto, por ser requisito de PyTorch/PyG para registrar
pesos entrenables) y `src/22_entrenar_gat_gcn.py` (entrenamiento con
validación cruzada K-fold estratificada por evento, un modelo
independiente por cohorte, no pooled).

Entrenado con éxito, con el mismo protocolo (20 épocas, 5 pliegues,
k=20 vecinos), sobre las 5 cohortes del TFM. Resultado real (C-index
medio ± desviación estándar entre pliegues):

BRCA 0,6255 ± 0,1301 · LUAD 0,5329 ± 0,0720 · LUSC 0,4514 ± 0,0642 ·
COAD 0,4599 ± 0,1107 · KIRC 0,4928 ± 0,0481

HALLAZGO: solo BRCA y LUAD superan el azar (0,5) con claridad; LUSC y
COAD quedan por debajo; KIRC es prácticamente indistinguible del azar.
El framework, tal como está implementado en este bloque (solo
GAT+GCN), NO generaliza de forma uniforme a las 5 cohortes. Dos
hipótesis razonadas, no confundibles entre sí ni verificadas: (a) el
diseño completo del TFM incluye VGAE/MIL/DEC precisamente para
capturar señal que GAT+GCN solos no alcanzan (el Bloque 6 evalúa solo
2 de las 5 técnicas previstas); (b) no se ha hecho ningún ajuste de
hiperparámetros por cohorte (arquitectura, épocas, k_vecinos, learning
rate fijos, elegidos una única vez a partir del perfil de eventos de
BRCA). Ver `results/2026-08-07-paso40/runlog.txt` para la discusión
completa comparando las 5 cohortes (incluye además el hallazgo de que
la relación "más eventos de validación → menor varianza del C-index"
es una tendencia general, no una regla estricta: COAD y KIRC la
rompen en direcciones opuestas).

LECCIONES TÉCNICAS a recordar para los próximos bloques (VGAE, MIL,
DEC):
- El GAT aplicado paciente a paciente en un bucle Python sobre un
  grafo de ~20.000 nodos, sin llamar a `backward()` hasta el final del
  batch, agota la memoria (OOM): PyTorch retiene el grafo de autograd
  de TODOS los pacientes del batch simultáneamente. Solución aplicada
  en `21_modelo_gat_gcn.py`: envolver la llamada al GAT por paciente en
  `torch.utils.checkpoint.checkpoint` (`use_reentrant=False`), que
  recalcula las activaciones durante el backward en vez de retenerlas,
  acotando el pico de memoria a un paciente en vez de a todo el batch.
- Los entrenamientos largos (15-20h por cohorte en este hardware, CPU
  sin GPU) se lanzan con `nohup ... & disown` para sobrevivir a
  desconexiones de terminal — necesario en esta sesión, con un
  historial real de varios cortes de conexión de WSL a mitad de
  entrenamiento. IMPORTANTE: usar siempre `python3 -u` (o
  `PYTHONUNBUFFERED=1`) al redirigir la salida a un fichero de log; sin
  esto, `print()` se bufferiza al no ser una TTY y el log queda vacío
  hasta que el proceso termina, inutilizando el log para verificar
  progreso en tiempo real (ver incidente en
  `results/2026-08-04-paso36/runlog.txt`).
- Al monitorizar un proceso en background con `tail -f --pid=$PID`, un
  corte transitorio de la VM de WSL puede matar el proceso de `tail`
  sin matar el proceso Python real, generando un falso aviso de
  "proceso terminado". Más robusto: sondear `kill -0 $PID` en un
  bucle, en vez de depender del seguimiento interno de `tail --pid`
  (ver incidente y corrección en `results/2026-08-05-paso37/runlog.txt`).

Estado del documento Word del TFM (`TFM_bioinfor_corregido.docx`, no
incluido en este repositorio): ya actualizado con los resultados
reales de este bloque en la Tabla 6 y la discusión de la Sección 5.6.
Las Secciones 5.3/5.4/5.5 (hipótesis H2/H3/H4) siguen pendientes de
MIL, DEC y el análisis de biomarcadores, respectivamente.

Bloque 7 (VGAE) — CERRADO (2026-08-09): implementado en
`src/23_modelo_vgae.py` (arquitectura Kipf y Welling 2016: encoder de
tronco GCN compartido + 2 cabezas GCN para mu/log-varianza, decoder
también GCN, sobre el grafo paciente-paciente k-NN ya usado en
GAT+GCN) y `src/24_entrenar_vgae.py` (pre-entrenamiento independiente,
ELBO con free bits). Investigación exhaustiva sobre BRCA con 6
hallazgos técnicos documentados con honestidad completa: NaN por
logvar sin acotar; KL collapse temprano sin annealing; smoke test
injusto por escalado proporcional de la máscara; posterior collapse
tardío (épocas 26-30) sin free bits (Kingma et al. 2016); falta de
reproducibilidad por semilla global de PyTorch no fijada por canal; y
un último intento de reducción de dimensionalidad (2.500 genes por
varianza + arquitectura más pequeña) que mejora pero no cruza el
umbral de "mejora clara" fijado de antemano. Ver
`results/2026-08-08-paso41/runlog.txt`, `results/2026-08-08-paso42/runlog.txt`
y `results/2026-08-09-paso43/runlog.txt` para el detalle completo.

RESULTADO FINAL (reproducible, semilla=0 por canal, diseño de 3 VGAE
independientes por canal — único diseño que supera la base trivial en
más de un canal, corrigiendo la variante inicial con los 3 canales
concatenados que nunca la superó en ningún intento): RNA-seq RMSE
0,9902 (+1,0% vs. base trivial), metilación 0,9877 (+1,2%), CNV 1,0137
(-1,4%, PEOR que la base trivial). Mejora marginal, no confirmada como
robusta (una sola semilla, sin recursos para probar más). Se
documenta como limitación metodológica honesta para la Sección 4.9
del TFM y NO se replica en las 4 cohortes restantes; se deja la puerta
abierta a retomarlo en el futuro con más semillas de validación o más
pacientes por cohorte.

Bloque 8 (MIL) — EN CURSO: prerrequisito completado, el modelo
GAT+GCN final de BRCA (entrenado con `src/25_entrenar_gat_gcn_final.py`,
sin partición k-fold) guardado en
`data/processed/BRCA_modelo_gat_gcn_final.pt`. Arquitectura MIL
implementada en `src/26_modelo_mil.py` (`EncoderResNetMIL`: ResNet-50
preentrenado en ImageNet, ver NOTA en el propio fichero sobre la
desviación respecto al checkpoint histología-específico que pide la
fórmula del TFM; `AtencionMIL`: mecanismo de atención de Ilse et al.
2018, Sección 4.5) y pipeline de procesamiento por paciente en
`src/27_procesar_wsi_paciente.py` (teselado 256x256 + filtrado de
tejido por saturación HSV + ResNet-50 + atención).

HALLAZGO Y CORRECCIÓN DE MEMORIA (paciente de prueba TCGA-A1-A0SB,
WSI de 741 MB): la primera versión de `27_procesar_wsi_paciente.py`
acumulaba todos los parches crudos con tejido en memoria antes de
convertirlos a tensor, lo que agotó la RAM de la máquina (5,7 GB) a
los 28.126 parches (63% del teselado), sin llegar a arrancar
ResNet-50 — inviable no solo para las 5 cohortes sino para UN solo
paciente. Corregido con procesamiento en streaming: el embedding de
cada lote de 16 parches se calcula en cuanto se completa el lote,
descartando los arrays de imagen crudos inmediatamente (solo se
acumulan en memoria los embeddings ya reducidos, ~8KB/parche frente a
~192KB/parche de los arrays crudos). Ver
`results/2026-08-10-paso44/runlog.txt` (intento fallido) y
`results/2026-08-10-paso45/runlog.txt` (corrección y ejecución
completa).

TIEMPO REAL VERIFICADO (mismo paciente, ya con el diseño en
streaming): 36.307 parches con tejido de 141.912 celdas en rejilla
completa (25,6% con tejido), procesados en 11.009,1s (~3h 3,5min,
teselado+ResNet-50 combinados) + 1,6s de atención MIL — total 183,5
min por paciente en este hardware (CPU, sin GPU, 2 núcleos). Pico de
RAM: 1.447 MB (frente a los >5.130 MB, sin terminar, del diseño
original). Con este dato real, la extrapolación al volumen completo de
las 5 cohortes (698 GB estimados) queda pendiente de decidir.

Trabajo pendiente: completar el Bloque 8 (MIL) — pendiente aún
descargar el resto de WSI de las 5 cohortes (solo el paciente de
prueba TCGA-A1-A0SB está descargado) y decidir la estrategia de
volumen completo a la luz del tiempo real verificado (~3h/paciente).
Después, DEC (Bloque 9, subtipos moleculares). STRING v12 ya está
descargado; Reactome y KEGG siguen pendientes para completar el grafo
heterogéneo de vías (Fase 3). `docs/manifest-datos.tsv` ya está al día
con el estado de Fase 1 (datos) pero NO refleja aún VGAE/MIL;
`README.md` tampoco — todavía dice "solo está descargada la cohorte
TCGA-BRCA" y no refleja las 5 cohortes completas ni los Bloques 6-8,
pendiente de actualizar.

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
