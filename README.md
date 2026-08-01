\# TFM HGNN-OmicSurv - Maria Cuenca Reyes

Master Universitario en Bioinformatica - UAX

## Objetivo

Este repositorio implementa el framework HGNN-OmicSurv sobre datos reales de TCGA, en dos fases, con el objetivo final de cubrir las 5 tecnicas de la Seccion 4 del TFM (GAT, GCN, VGAE, MIL, DEC) sobre las 5 cohortes de TCGA (BRCA, LUAD, LUSC, COAD, KIRC):

**Fase 1 (completada para las 5 cohortes):** pipeline de adquisicion, validacion y preprocesamiento de datos reales del TCGA. Verificado con datos reales de las cohortes TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-COAD y TCGA-KIRC en sus cuatro capas omicas: datos clinicos completos (numero de pacientes distinto por cohorte, ver Tabla 2 del TFM
y docs/manifest-datos.tsv), y muestra representativa de RNA-seq, CNV y metilacion (n=200 pacientes cada una, coordinados por case_id y con muestra tumoral primaria consistente entre las 3 modalidades, todas las cohortes con el mismo array de metilacion Illumina 450K, homogeneo entre cohortes; ver "Coordinacion de pacientes entre modalidades" mas abajo).

**Fase 2 (en curso):** implementacion y entrenamiento de las cinco arquitecturas del framework —GAT (Graph Attention Network), GCN (Graph Convolutional
Network), VGAE (Variational Graph Autoencoder), MIL (Multiple Instance Learning con atencion) y DEC (Deep Embedded Clustering)— sobre PyTorch Geometric,
usando los datos verificados en la Fase 1, para la prediccion de supervivencia, el manejo generativo de datos faltantes, la integracion de imagenes WSI
y el descubrimiento de subtipos moleculares en pacientes de cancer. Actualmente en desarrollo: GAT, GCN y VGAE sobre TCGA-BRCA; MIL y DEC pendientes de implementar.

## Estructura del proyecto
data/raw/           Datos originales descargados (GDC, STRING)
data/processed/     Datos generados por los scripts (validados, preprocesados)
src/                Scripts numerados en orden de ejecucion (00_, 01_, ...)
results/            Runlogs y resultados de cada paso del proceso
docs/               Manifest de datos y tests
config/             Ficheros de configuracion
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
- Dependencias adicionales para MIL (lectura de WSI) y DEC: pendientes de definir e incorporar a este fichero

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



\## Idempotencia

Si se ejecuta el mismo script dos veces seguidas sin modificar los ficheros de entrada, el pipeline lo detecta mediante el hash SHA-256 guardado en
logs/manifest.json y omite el recalculo.



\## Trazabilidad de ejecuciones

Cada ejecucion relevante se documenta en results/<fecha>-expNN/runlog.txt con: ID de ejecucion, comando exacto, commit de Git correspondiente y
observaciones.



