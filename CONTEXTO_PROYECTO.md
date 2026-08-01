# Contexto del proyecto - TFM HGNN-OmicSurv

## Que es este proyecto
Trabajo Fin de Master en Bioinformatica (UAX). Framework de integracion 
multi-omica con Redes Neuronales de Grafos Heterogeneas para prediccion 
de supervivencia oncologica, usando datos del TCGA.

## Alcance objetivo (IMPORTANTE)
El proyecto debe implementar las 5 tecnicas descritas en la Seccion 4 
del TFM (documento Word ya redactado, no incluido en este repositorio) 
sobre las 5 cohortes de TCGA:

Tecnicas:
1. GAT (Graph Attention Network) - capa de atencion sobre genes
2. GCN (Graph Convolutional Network) - capa de propagacion entre pacientes
3. VGAE (Variational Graph Autoencoder) - imputacion generativa de datos faltantes
4. MIL (Multiple Instance Learning con atencion) - integracion de imagenes WSI
5. DEC (Deep Embedded Clustering) - descubrimiento de subtipos moleculares

Cohortes TCGA:
- BRCA (cancer de mama)
- LUAD (adenocarcinoma de pulmon)
- LUSC (carcinoma escamoso de pulmon)
- COAD (cancer de colon)
- KIRC (cancer de rinon)

## Estado actual del repositorio (punto de partida, NO empezar de cero)

### Ya completado y verificado (Fase 1 - pipeline de datos, solo BRCA):
- src/00_generar_datos_prueba.py - genera datos ficticios de prueba
- src/01_validate_data.py - valida datos clinicos y omicos
- src/02_filtrar_manifest.py, 02b, 02c - filtran manifest del GDC por tipo de dato
- src/03_convertir_rnaseq.py - convierte RNA-seq real descargado a tabla unica
- src/04_convertir_clinical.py - convierte datos clinicos reales del GDC
- src/05_estadisticas_descriptivas.py - estadisticas reales sobre BRCA
- src/06_grafico_supervivencia.py - histograma real de supervivencia
- src/10_convertir_cnv.py - convierte CNV real
- src/11_convertir_metilacion.py - convierte metilacion real
- src/12_prueba_pytorch_geometric.py - prueba minima de PyTorch Geometric

Datos reales ya descargados en data/raw/ (SOLO cohorte BRCA):
- BRCA_clinical.tsv (1098 pacientes reales completos)
- BRCA_rnaseq.tsv (muestra de 200 pacientes reales, coordinados con CNV y metilacion)
- BRCA_cnv.tsv (muestra de 200 pacientes reales, coordinados con RNA-seq y metilacion)
- BRCA_metilacion.tsv (muestra de 200 pacientes reales, coordinados con RNA-seq y CNV)
- 9606.protein.links.v12.0.txt.gz (STRING v12 completo)
- gen_proteina_conversion.txt (tabla ENSG->ENSP de BioMart)

NOTA (Paso 24, 2026-07-31): los 20 pacientes originales de las 3 capas
omicas de BRCA NO estaban coordinados entre si (cada modalidad se
selecciono de forma independiente; al cruzarlas solo 1 paciente
coincidia en las 3). Corregido ampliando a 200 pacientes tomados de la
interseccion de 786 candidatos con las 3 modalidades disponibles. Esto
tambien obligo a reescribir src/11_convertir_metilacion.py a
procesamiento en streaming porque la version anterior agotaba la
memoria ("Killed") al procesar 486.427 sitios CpG x 200 pacientes.
Ver results/2026-07-31-paso24/runlog.txt para el detalle completo.

NOTA (Paso 25, 2026-07-31): misma correccion replicada en LUAD
(LUAD_rnaseq.tsv, LUAD_cnv.tsv, LUAD_metilacion.tsv ya con 200
pacientes coordinados). A diferencia de BRCA, el manifest de LUAD no
trae el case ID del paciente en el nombre de fichero de RNA-seq ni
metilacion, asi que se resolvio consultando la API del GDC
(cases.submitter_id y cases.samples.sample_type por file_id); ademas
se descubrio que RNA-seq y metilacion (no solo CNV) tambien pueden
tener mas de un fichero candidato por paciente (tumor primario vs.
tejido normal/recurrente), resuelto filtrando a "Primary Tumor" y, si
aun asi quedaba empate, con desempate final deterministico por
file_id. Interseccion real con las 3 modalidades: 454 casos (de los
que se tomaron los primeros 200 alfabeticamente). Ver
results/2026-07-31-paso25/runlog.txt para el detalle completo.

NOTA (Paso 26, 2026-08-01): misma correccion replicada en LUSC
(LUSC_rnaseq.tsv, LUSC_cnv.tsv, LUSC_metilacion.tsv ya con 200
pacientes coordinados), aplicando esta vez desde el principio el
criterio completo aprendido en LUAD (consulta a la API del GDC, filtro
"Primary Tumor" y desempate final por file_id en las 3 modalidades, no
solo en CNV). Interseccion real con las 3 modalidades: 368 casos (de
los que se tomaron los primeros 200 alfabeticamente); confirmado con
el usuario antes de proceder a la descarga. Ver
results/2026-08-01-paso26/runlog.txt para el detalle completo.

NOTA (Paso 27, 2026-08-01): misma correccion replicada en COAD
(COAD_rnaseq.tsv, COAD_cnv.tsv, COAD_metilacion.tsv ya con 200
pacientes coordinados), aplicando el criterio completo (API del GDC,
filtro "Primary Tumor", desempate por file_id en las 3 modalidades)
desde el principio. Interseccion real con las 3 modalidades: 295
casos (de los que se tomaron los primeros 200 alfabeticamente);
margen mas ajustado que en BRCA/LUAD/LUSC. Ver
results/2026-08-01-paso27/runlog.txt para el detalle completo.

NOTA (Paso 28, 2026-08-01): misma correccion replicada en KIRC
(ultima de las 5 cohortes), con el mismo criterio completo desde el
principio. Interseccion real con las 3 modalidades: 316 casos (de los
que se tomaron los primeros 200 alfabeticamente); margen de 116
pacientes, mas holgado que COAD. Ver results/2026-08-01-paso28/runlog.txt
para el detalle completo.

NOTA (Paso 29, 2026-08-01): al cerrar el bloque de las 5 cohortes, se
verifico BRCA con el mismo metodo riguroso usado en LUAD-KIRC
(traducir cada UUID de columna a case_id y sample_type via la API del
GDC), no aplicado hasta ahora sobre BRCA porque esa cohorte se
corrigio (Paso 24) ANTES de introducir el filtro "Primary Tumor"
(introducido despues, en LUAD, Paso 25). Resultado: 10 de los 200
pacientes de BRCA (5%) tenian tipo de muestra inconsistente entre
modalidades (ej. metilacion de tejido normal en vez de tumoral; 3
casos con DOS de las 3 modalidades siendo tejido normal). Corregido
sustituyendo esos 10 pacientes por los siguientes 10 candidatos
validos (Primary Tumor consistente) de los 779 candidatos rigurosos de
BRCA (cifra que sustituye a los "786" citados mas abajo, calculados
sin el filtro Primary Tumor), sin tocar a los otros 190 pacientes ya
correctos. Ver results/2026-08-01-paso29/runlog.txt para la tabla
completa de los 10 casos y el detalle de la correccion.

RESUMEN: con el Paso 29 se completa, en las 5 cohortes del TFM, la
coordinacion de pacientes entre las 3 capas omicas principales
(RNA-seq, CNV, metilacion) CON muestra tumoral primaria consistente.
Cada cohorte tiene 200 pacientes que comparten exactamente el mismo
case_id en las 3 tablas, y los 200 tienen "Primary Tumor" en las 3
modalidades simultaneamente (verificado explicitamente en cada
cohorte, traduciendo cada columna/UUID de fichero a su case_id y
sample_type real, no solo asumido):

  Cohorte | Candidatos (3 modalidades, Primary Tumor) | Runlog
  BRCA    | 779 (corregido en Paso 29)                | results/2026-07-31-paso24/runlog.txt, results/2026-08-01-paso29/runlog.txt
  LUAD    | 454                                        | results/2026-07-31-paso25/runlog.txt
  LUSC    | 368                                        | results/2026-08-01-paso26/runlog.txt
  COAD    | 295                                        | results/2026-08-01-paso27/runlog.txt
  KIRC    | 316                                        | results/2026-08-01-paso28/runlog.txt

Pendiente (fuera del alcance de este bloque de trabajo): valorar si
conviene consolidar en un script de src/ el proceso ad-hoc usado para
identificar y descargar los pacientes coordinados (consulta a la API
del GDC + filtro Primary Tumor + desempate por file_id), por si hace
falta repetirlo (p. ej. ampliar la muestra mas alla de 200 pacientes
en el futuro).

### Scripts recuperados pero SIN adaptar al nuevo alcance (generaban figuras ilustrativas con datos de ejemplo, no datos reales):
- src/07_figura4_cindex.py
- src/08_figura6_subtipos_paad.py
- src/09_figura7_roc.py

### Pendiente de implementar (esto es el trabajo a realizar):
- Descargar y procesar las 4 cohortes restantes (LUAD, LUSC, COAD, KIRC), 
  replicando el pipeline ya usado en BRCA
- Descargar Reactome y KEGG (STRING ya esta descargado)
- Construir el grafo heterogeneo real (genes, proteinas, metabolitos, vias)
- Implementar y entrenar GAT + GCN sobre PyTorch Geometric
- Implementar y entrenar VGAE
- Descargar WSI e implementar MIL
- Implementar DEC
- Generar todas las tablas y figuras de resultados con datos reales

## Entorno tecnico
- Python del sistema (Windows): 3.14, usado para Fase 1 (csv, matplotlib)
- Entorno virtual venv_pytorch (Python 3.11): PyTorch 2.13 CPU, 
  PyTorch Geometric 2.8 - usado para Fase 2 en adelante
- Sin GPU disponible (todo el entrenamiento sera sobre CPU)
- El README.md y docs/manifest-datos.tsv actuales reflejan una decision 
  de alcance anterior (solo 3 tecnicas, solo BRCA) que debe actualizarse 
  para reflejar el alcance completo de 5 tecnicas y 5 cohortes

## Estilo de codigo a mantener
- Scripts simples y lineales, sin clases, funciones basicas
- Uso de csv estandar de Python (no pandas) para el pipeline de datos
- Cada script numerado secuencialmente en src/
- Cada paso documentado en results/<fecha>-pasoN/runlog.txt con: 
  ID de ejecucion, comando exacto, resultado obtenido, commit de Git
- Comentarios y nombres de variable en español, sin tildes en el codigo
