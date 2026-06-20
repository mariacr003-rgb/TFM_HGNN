\# TFM HGNN-OmicSurv - Maria Cuenca Reyes

Master Universitario en Bioinformatica - UAX

## Objetivo

Este repositorio implementa el pipeline de adquisicion, validacion y preprocesamiento de datos reales del TCGA (Seccion 3 del TFM), que
constituye la base de datos verificada sobre la que se proyecta la implementacion del framework HGNN-OmicSurv (Red Neuronal de Grafos
Heterogenea + VGAE + MIL + Deep Clustering, Seccion 4 del TFM) para prediccion de supervivencia, estratificacion histomolecular,
descubrimiento de subtipos moleculares e identificacion de biomarcadores de inmunoterapia.

El pipeline aqui contenido ha sido verificado con datos reales de la cohorte TCGA-BRCA en sus cuatro capas omicas (datos clinicos completos,
n=1.098; muestra representativa de RNA-seq, CNV y metilacion, n=20 cada una). La implementacion del framework HGNN-OmicSurv en si misma
(arquitecturas GAT, GCN, VGAE, MIL, DEC sobre PyTorch Geometric) y su ejecucion sobre las cinco cohortes completas (BRCA, LUAD, LUSC, COAD,
KIRC) constituyen la siguiente fase del proyecto, no implementada en el presente repositorio.

\## Estructura del proyecto

```

data/raw/           Datos originales descargados (GDC, STRING, Reactome, KEGG)

data/processed/   Datos generados por los scripts (validados, preprocesados)

scripts/             Scripts numerados en orden de ejecucion (01\_, 02\_, ...)

results/          Tablas y figuras finales (Seccion 5 del TFM)

docs/                Manifest de datos y tests

config/              Ficheros de configuracion (config.yaml)

entorno/             Especificacion del entorno (requirements.txt)

```



\## Entrada de prueba

docs/tests/test_clinical.tsv y docs/tests/test_rnaseq.tsv (mini dataset simulado de 10 pacientes, generado con src/00_generar_datos_prueba.py, para verificar que el pipeline funciona antes de usar datos reales del TCGA.

IMPORTANTE: al ejecutar src/01_validate_data.py sobre estos datos de prueba, se debe usar la etiqueta de cohorte "PRUEBA", para no confundir los resultados de prueba con los resultados reales en data/processed/. Ver docs/manifest-datos.tsv para el detalle de cada fichero.

\## Entrada real

data/raw/BRCA_clinical.tsv y data/raw/BRCA_rnaseq.tsv (descargados del portal GDC, ver docs/manifest-datos.tsv para el origen exacto y la URL).

\## Dependencias

\- Python 3.10
\- pandas, numpy (Paso 1)
\- PyTorch 2.1, PyTorch Geometric 2.4 (pasos posteriores de la Seccion 4)
\- Git



\## Como ejecutar (Paso 1)

Con datos de prueba:

```
python src/00_generar_datos_prueba.py docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv

python src/01_validate_data.py PRUEBA docs/tests/test_clinical.tsv docs/tests/test_rnaseq.tsv data/processed
```



Con datos reales:

```

python src/01_validate_data.py BRCA data/raw/BRCA_clinical.tsv data/raw/BRCA_rnaseq.tsv data/processed

```



\## Idempotencia

Si se ejecuta el mismo script dos veces seguidas sin modificar los ficheros de entrada, el pipeline lo detecta mediante el hash SHA-256 guardado en
logs/manifest.json y omite el recalculo.



\## Trazabilidad de ejecuciones

Cada ejecucion relevante se documenta en resultados/<fecha>-expNN/runlog.txt con: ID de ejecucion, comando exacto, commit de Git correspondiente y
observaciones.



