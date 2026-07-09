\# TFM HGNN-OmicSurv - Maria Cuenca Reyes

Master Universitario en Bioinformatica - UAX

## Objetivo

Este repositorio implementa el framework HGNN-OmicSurv sobre datos reales de TCGA-BRCA, en dos fases:

**Fase 1 (completada):** pipeline de adquisicion, validacion y preprocesamiento de datos reales del TCGA. Verificado con datos reales de la cohorte TCGA-BRCA en sus cuatro capas omicas (datos clinicos completos, n=1.098; muestra representativa de RNA-seq,
CNV y metilacion, n=20 cada una).

**Fase 2 (en curso):** implementacion y entrenamiento de tres arquitecturas del framework —GAT (Graph Attention Network), GCN (Graph Convolutional
Network) y VGAE (Variational Graph Autoencoder)— sobre PyTorch Geometric, usando los datos verificados en la Fase 1, para la prediccion de
supervivencia y el manejo generativo de datos faltantes en pacientes de cancer de mama (TCGA-BRCA).

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
data/raw/BRCA_clinical.tsv, data/raw/BRCA_rnaseq.tsv, data/raw/BRCA_cnv.tsv,
data/raw/BRCA_metilacion.tsv (descargados del portal GDC), y
data/raw/9606.protein.links.v12.0.txt.gz (descargado de STRING v12).
Ver docs/manifest-datos.tsv para el origen exacto y la URL de cada fichero.

## Dependencias

### Fase 1 (pipeline de datos)
- Python 3.14 
- Modulos estandar: csv, sys, pathlib
- matplotlib (figuras de la Seccion 5)

### Fase 2 (framework GAT+GCN+VGAE)
- Python 3.11 (entorno virtual independiente: venv_pytorch/)
- PyTorch 2.13 (CPU)
- PyTorch Geometric 2.8

## Como ejecutar

### Fase 1 - Pipeline de datos
Con datos de prueba:

python src/00_generar_datos_prueba.py docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv
python src/01_validate_data.py PRUEBA docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv data/processed

Con datos reales:

python src/01_validate_data.py BRCA data/raw/BRCA_clinical.tsv data/raw/BRCA_rnaseq.tsv data/processed

### Fase 2 - Framework GAT+GCN+VGAE
venv_pytorch\Scripts\activate
python src/12_prueba_pytorch_geometric.py



\## Idempotencia

Si se ejecuta el mismo script dos veces seguidas sin modificar los ficheros de entrada, el pipeline lo detecta mediante el hash SHA-256 guardado en
logs/manifest.json y omite el recalculo.



\## Trazabilidad de ejecuciones

Cada ejecucion relevante se documenta en results/<fecha>-expNN/runlog.txt con: ID de ejecucion, comando exacto, commit de Git correspondiente y
observaciones.



