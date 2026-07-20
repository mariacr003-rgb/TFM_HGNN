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

**Estado actual: solo BRCA está descargada y solo GAT+GCN+VGAE están en
desarrollo** (Fase 2). LUAD/LUSC/COAD/KIRC, MIL y DEC son trabajo
pendiente. El `README.md` y `docs/manifest-datos.tsv` reflejan todavía
el alcance anterior (solo BRCA, solo 3 técnicas) y están pendientes de
actualizar al alcance completo — no asumir que documentan el estado
final.

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
  fichero, para elegir qué descargar con `gdc-client.exe`
- `03_convertir_rnaseq.py`, `04_convertir_clinical.py`,
  `10_convertir_cnv.py`, `11_convertir_metilacion.py` → convierten los
  ficheros crudos descargados del GDC (uno por paciente, en
  subcarpetas con UUID) a una tabla única por cohorte en `data/raw/`
- `05_estadisticas_descriptivas.py`, `06_grafico_supervivencia.py` →
  estadísticas y figuras reales sobre los datos validados
- `07_figura4_cindex.py`, `08_figura6_subtipos_paad.py`,
  `09_figura7_roc.py` → generan figuras de resultados, pero **todavía
  con datos de ejemplo ilustrativos, sin adaptar a datos reales**
- `12_prueba_pytorch_geometric.py` → primer script de Fase 2 (PyG)

IMPORTANTE — convención de nombres de los manifests completos en
`data/raw/`: el de BRCA se llama `gdc_manifest_full.txt`, **sin**
prefijo de cohorte, porque fue el primero que se descargó (manualmente,
desde el botón "Manifest" del portal). Los de las 4 cohortes
siguientes, generadas vía API del GDC (ver
`results/2026-07-17-paso6/runlog.txt`), sí llevan prefijo de cohorte en
minúsculas: `luad_manifest_full.txt`, `lusc_manifest_full.txt`,
`coad_manifest_full.txt`, `kirc_manifest_full.txt`. No asumir que
`gdc_manifest_full.txt` es genérico o que existe un
`brca_manifest_full.txt` — no existe.

Flujo de datos: `data/raw/` (descargas GDC/STRING, muchas no
versionadas por tamaño — ver `.gitignore`) → `data/processed/` (salida
de validación, por cohorte: `<COHORTE>_clinical_valid.tsv`,
`<COHORTE>_rnaseq_valid.tsv`, `<COHORTE>_validation_report.txt`) →
`results/<fecha>-pasoN/` (figuras, tablas, runlog).

`docs/manifest-datos.tsv` documenta origen, rol y URL de cada fichero
de datos (hoy solo cubre BRCA + STRING; Reactome/KEGG y las otras 4
cohortes están pendientes de añadir ahí cuando se incorporen).

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
