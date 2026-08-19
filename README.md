\# TFM HGNN-OmicSurv - Maria Cuenca Reyes

Master Universitario en Bioinformatica - UAX

## Objetivo

Este repositorio implementa el framework HGNN-OmicSurv sobre datos reales de TCGA, en dos fases, con el objetivo final de cubrir las 5 tecnicas de la Seccion 4 del TFM (GAT, GCN, VGAE, MIL, DEC) sobre las 5 cohortes de TCGA (BRCA, LUAD, LUSC, COAD, KIRC):

**Fase 1 (completada para las 5 cohortes):** pipeline de adquisicion, validacion y preprocesamiento de datos reales del TCGA. Verificado con datos reales de las cohortes TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-COAD y TCGA-KIRC en sus cuatro capas omicas: datos clinicos completos (numero de pacientes distinto por cohorte, ver Tabla 2 del TFM
y docs/manifest-datos.tsv), y muestra representativa de RNA-seq, CNV y metilacion (n=200 pacientes cada una, coordinados por case_id y con muestra tumoral primaria consistente entre las 3 modalidades, todas las cohortes con el mismo array de metilacion Illumina 450K, homogeneo entre cohortes; ver "Coordinacion de pacientes entre modalidades" mas abajo).

**Fase 2 (en curso):** implementacion y entrenamiento de las cinco arquitecturas del framework —GAT (Graph Attention Network), GCN (Graph Convolutional
Network), VGAE (Variational Graph Autoencoder), MIL (Multiple Instance Learning con atencion) y DEC (Deep Embedded Clustering)— sobre PyTorch Geometric,
usando los datos verificados en la Fase 1, para la prediccion de supervivencia, el manejo generativo de datos faltantes, la integracion de imagenes WSI
y el descubrimiento de subtipos moleculares en pacientes de cancer. Estado real (actualizado 2026-08-18, ver CLAUDE.md para el detalle completo):

- **GAT+GCN (Bloque 6) — CERRADO**, entrenado y evaluado con validacion cruzada en las 5 cohortes.
- **VGAE (Bloque 7) — CERRADO**, investigado a fondo sobre BRCA; mejora marginal sobre la base trivial, no confirmada como robusta (limitacion metodologica documentada, no replicado en las 4 cohortes restantes).
- **MIL (Bloque 8) — CERRADO en las 5 cohortes** (2026-08-19): descarga+procesamiento de WSI completo (50/50 pacientes, 0 fallos) y C-index de MIL+molecular calculado; limitacion metodologica documentada (sobreajuste severo con solo 10 pacientes/cohorte, ver resultados abajo).
- **DEC (Bloque 9) — CERRADO en las 5 cohortes**: unico resultado no degenerado es K=2 (limitacion metodologica confirmada y generalizada, no el descubrimiento de subtipos multiples buscado originalmente).

## Estructura del proyecto
data/raw/           Datos originales descargados (GDC, STRING)
data/processed/     Datos generados por los scripts (validados, preprocesados)
src/                Scripts numerados en orden de ejecucion (00_, 01_, ...)
results/            Runlogs y resultados de cada paso del proceso
docs/               Manifest de datos y tests
entorno/            Especificacion del entorno (requirements.txt)
venv_pytorch/       Entorno virtual Python 3.11 con PyTorch (no versionado)

## Entrada de prueba
docs/tests/test_clinical.tsv y docs/tests/test_rnaseq.tsv (mini dataset
simulado de 10 pacientes, generado con src/00_generar_datos_prueba.py,
para verificar que el pipeline funciona antes de usar datos reales).

IMPORTANTE: al ejecutar src/01_validate_data.py sobre estos datos de
prueba, se debe usar la etiqueta de cohorte "PRUEBA", para no confundir
los resultados de prueba con los resultados reales en data/processed/.

## Entrada real
Las 5 cohortes (BRCA, LUAD, LUSC, COAD, KIRC) tienen ya descargadas y
verificadas sus 4 capas de datos reales: data/raw/<COHORTE>_clinical.tsv,
data/raw/<COHORTE>_rnaseq.tsv, data/raw/<COHORTE>_cnv.tsv,
data/raw/<COHORTE>_metilacion.tsv (descargados del portal GDC), mas
data/raw/9606.protein.links.v12.0.txt.gz (descargado de STRING v12).

La metilacion de las 5 cohortes usa el mismo array Illumina 450K
(486.427 sitios CpG), estandarizado tras detectar y corregir una
mezcla de arrays incompatibles (27K/450K/EPIC) en la muestra inicial
de BRCA y LUAD (ver results/2026-07-24-paso11/runlog.txt).

Reactome y KEGG estan pendientes de descargar para completar el grafo
heterogeneo de vias (Fase 3).
Ver docs/manifest-datos.tsv para el origen exacto y la URL de cada fichero.

## Coordinacion de pacientes entre modalidades

RNA-seq, CNV y metilacion se descargan por separado del GDC, y el
nombre de fichero no incluye el identificador del paciente (case_id) en
RNA-seq ni en metilacion. Para que las 3 tablas de una cohorte
contengan exactamente los mismos 200 pacientes, y que sean muestra
tumoral primaria ("Primary Tumor") en las 3 modalidades y no una
mezcla de tumor/tejido normal del mismo paciente, se usan 4 scripts que
consultan la API REST del GDC (https://api.gdc.cancer.gov/files):

- src/14_mapear_case_id.py: resuelve, para una lista de ficheros
  candidatos (file_id + filename), su case_id y tipo de muestra
  (sample_type) via la API del GDC.
- src/15_interseccion_pacientes.py: a partir de la salida del script
  anterior para las 3 modalidades, filtra a muestra "Primary Tumor",
  desempata los casos con mas de un fichero candidato (prioridad de
  pipeline en CNV, despues file_id menor) y calcula la interseccion de
  pacientes con las 3 modalidades disponibles; puede guardar la lista
  completa o solo los primeros N casos.
- src/16_generar_manifests_coordinados.py: genera los 3 manifests en
  formato GDC (descargables con gdc-client.exe) a partir de la
  seleccion de pacientes anterior.
- src/17_verificar_coordinacion.py: verifica, sobre las 3 tablas
  finales ya generadas, que comparten exactamente los mismos pacientes
  Y que los 200 tienen muestra "Primary Tumor" consistente en las 3
  modalidades (vuelve a consultar la API, no se fia de ningun fichero
  de seleccion previo).

Ver results/2026-07-31-paso24/runlog.txt (deteccion del problema y
primera correccion en BRCA), results/2026-07-31-paso25/runlog.txt a
results/2026-08-01-paso28/runlog.txt (replicacion en LUAD, LUSC, COAD,
KIRC), results/2026-08-01-paso29/runlog.txt (correccion de 10
pacientes de BRCA con tipo de muestra inconsistente) y
results/2026-08-01-paso30/runlog.txt (consolidacion de estos 4
scripts, antes ad-hoc) para el detalle completo de este bloque de
trabajo.

## Dependencias

### Fase 1 (pipeline de datos)
- Python 3.14 
- Modulos estandar: csv, sys, pathlib
- matplotlib (figuras de la Seccion 5)

### Fase 2 (framework GAT+GCN+VGAE+MIL+DEC)
- Python 3.11 (entorno virtual independiente: venv_pytorch/)
- PyTorch 2.13 (CPU)
- PyTorch Geometric 2.8
- torchvision 0.28 (ResNet-50 preentrenado en ImageNet, encoder de MIL)
- openslide-python 1.4.6 + openslide-bin 4.0.1.2 (lectura de WSI .svs para MIL)
- DEC (Bloque 9) no añade dependencias nuevas: reutiliza directamente el embedding ya calculado por GAT+GCN (`h_final`), sin autoencoder propio

Todas las dependencias de Fase 2 estan fijadas en `entorno/requirements.txt`.

## Como ejecutar

### Fase 1 - Pipeline de datos
Con datos de prueba:

python src/00_generar_datos_prueba.py docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv
python src/01_validate_data.py PRUEBA docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv data/processed

Con datos reales:

python src/01_validate_data.py BRCA data/raw/BRCA_clinical.tsv data/raw/BRCA_rnaseq.tsv data/processed

### Fase 2 - Framework GAT+GCN+VGAE+MIL+DEC
venv_pytorch\Scripts\activate
python src/12_prueba_pytorch_geometric.py

Scripts principales de cada bloque (todos numerados en `src/`, ejecutables de forma independiente):

- `19_proyectar_grafo_gen_gen.py`, `20_preprocesar_atributos_gen.py`, `21_modelo_gat_gcn.py`, `22_entrenar_gat_gcn.py` — Bloque 6 (GAT+GCN): grafo gen-gen, atributos por gen, modelo y entrenamiento con validacion cruzada.
- `23_modelo_vgae.py`, `24_entrenar_vgae.py` — Bloque 7 (VGAE): imputacion generativa de datos faltantes.
- `25_entrenar_gat_gcn_final.py` — modelo GAT+GCN final por cohorte (sin pliegues), prerrequisito de MIL y DEC.
- `26_modelo_mil.py`, `27_procesar_wsi_paciente.py`, `30_seleccionar_pacientes_mil.py`, `31_descargar_procesar_mil.py` — Bloque 8 (MIL): arquitectura, procesamiento de WSI por paciente en streaming, seleccion de pacientes y orquestacion de descarga+procesamiento con reintento ante fallos de red.
- `28_entrenar_dec.py`, `29_barrido_dec.py` — Bloque 9 (DEC): clustering sobre el embedding de GAT+GCN, con barrido de hiperparametros.

Ejemplo de uso (barrido de DEC sobre una cohorte ya entrenada):

python src/29_barrido_dec.py BRCA data/processed/BRCA_modelo_gat_gcn_final.pt data/processed/BRCA_dec_subtipos.pt

## Resultados reales (actualizado 2026-08-18)

**GAT+GCN (Bloque 6), C-index medio ± desviacion estandar entre pliegues, validacion cruzada 5-fold:**

| Cohorte | C-index |
|---|---|
| BRCA | 0,6255 ± 0,1301 |
| LUAD | 0,5329 ± 0,0720 |
| LUSC | 0,4514 ± 0,0642 |
| COAD | 0,4599 ± 0,1107 |
| KIRC | 0,4928 ± 0,0481 |

Solo BRCA y LUAD superan el azar (0,5) con claridad; ver CLAUDE.md (Paso 40) para la discusion completa.

**DEC (Bloque 9), K=2 como unico resultado no degenerado en las 5 cohortes (entropia normalizada, 0=colapsado, 1=perfectamente equilibrado):**

| Cohorte | Entropia | Reparto de clusters |
|---|---|---|
| BRCA | 0,937 | [64, 117] |
| LUAD | 0,906 | [114, 54] |
| LUSC | 0,992 | [102, 83] |
| COAD | 0,950 | [113, 66] |
| KIRC | 0,945 | [66, 116] |

Cualquier K≥3 colapsa de forma reproducible en las 5 cohortes (ver CLAUDE.md, Bloque 9, para el detalle del hallazgo y la correccion del criterio de decision).

**MIL+molecular (Bloque 8), C-index sobre 10 pacientes/cohorte (sin holdout - ver advertencia abajo), comparado con el C-index solo molecular del Bloque 6:**

| Cohorte | C-index MIL+molecular | Pares comparables | C-index solo molecular (Bloque 6) |
|---|---|---|---|
| BRCA | no calculable (0 eventos) | 0/90 | 0,6255 |
| LUAD | 0,9286 | 14/90 | 0,5329 |
| LUSC | 0,8276 | 29/90 | 0,4514 |
| COAD | 0,9048 | 21/90 | 0,4599 |
| KIRC | 1,0000 | 4/90 | 0,4928 |

**ADVERTENCIA (leer antes de interpretar esta tabla):** la cabeza de riesgo tiene 2.081 parametros para solo 10 pacientes por cohorte (~208 parametros/muestra) - el C-index de MIL+molecular es una metrica de ajuste al propio conjunto de entrenamiento, NO de generalizacion, y casi con toda seguridad refleja sobreajuste extremo (el caso KIRC, C-index=1,0000 sobre solo 4 pares comparables con 1 unico evento, es el ejemplo mas claro). NO es evidencia de que MIL mejore la prediccion de supervivencia frente al Bloque 6, y los dos C-index de esta tabla NO son directamente comparables (protocolos de evaluacion distintos). Se documenta como limitacion metodologica honesta, ver CLAUDE.md (Bloque 8) y results/2026-08-19-paso53/runlog.txt para el detalle completo.



\## Idempotencia

Si se ejecuta el mismo script dos veces seguidas sin modificar los ficheros de entrada, el pipeline lo detecta mediante el hash SHA-256 guardado en
logs/manifest.json y omite el recalculo.



\## Trazabilidad de ejecuciones

Cada ejecucion relevante se documenta en results/<fecha>-expNN/runlog.txt con: ID de ejecucion, comando exacto, commit de Git correspondiente y
observaciones.



