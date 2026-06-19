\# TFM HGNN-OmicSurv - Maria Cuenca Reyes



Master Universitario en Bioinformatica - UAX



\## Objetivo

Implementar el framework HGNN-OmicSurv (Red Neuronal de Grafos Heterogenea +

VGAE + MIL + Deep Clustering) descrito en la Seccion 4 del TFM, para

prediccion de supervivencia, estratificacion histomolecular, descubrimiento

de subtipos moleculares e identificacion de biomarcadores de inmunoterapia

sobre datos del TCGA.



\## Estructura del proyecto

```

datos-raw/           Datos originales descargados (GDC, STRING, Reactome, KEGG)

datos-intermedios/   Datos generados por los scripts (validados, preprocesados)

scripts/             Scripts numerados en orden de ejecucion (01\_, 02\_, ...)

resultados/          Tablas y figuras finales (Seccion 5 del TFM)

docs/                Manifest de datos y tests

config/              Ficheros de configuracion (config.yaml)

entorno/             Especificacion del entorno (requirements.txt)

```



\## Entrada de prueba

docs/tests/test\_clinical.tsv y docs/tests/test\_rnaseq.tsv (mini dataset

simulado de 10 pacientes para verificar que el pipeline funciona antes de

usar datos reales del TCGA).



\## Entrada real

datos-raw/BRCA\_clinical.tsv y datos-raw/BRCA\_rnaseq.tsv (descargados del

portal GDC, ver docs/manifest-datos.tsv para el origen exacto y la URL).



\## Dependencias

\- Python 3.10

\- pandas, numpy (Paso 1)

\- PyTorch 2.1, PyTorch Geometric 2.4 (pasos posteriores de la Seccion 4)

\- Git



\## Como ejecutar (Paso 1)

Con datos de prueba:

```

python scripts/01\_validate\_data.py BRCA docs/tests/test\_clinical.tsv docs/tests/test\_rnaseq.tsv datos-intermedios

```



Con datos reales:

```

python scripts/01\_validate\_data.py BRCA datos-raw/BRCA\_clinical.tsv datos-raw/BRCA\_rnaseq.tsv datos-intermedios

```



\## Idempotencia

Si se ejecuta el mismo script dos veces seguidas sin modificar los ficheros

de entrada, el pipeline lo detecta mediante el hash SHA-256 guardado en

logs/manifest.json y omite el recalculo.



\## Trazabilidad de ejecuciones

Cada ejecucion relevante se documenta en resultados/<fecha>-expNN/runlog.txt

con: ID de ejecucion, comando exacto, commit de Git correspondiente y

observaciones, siguiendo el mismo formato adoptado en la Actividad UD2.



