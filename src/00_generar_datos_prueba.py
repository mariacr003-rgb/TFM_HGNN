"""
generar_datos_prueba.py
Genera datos de prueba ficticios para probar el pipeline antes de usar
datos reales del TCGA. No son datos reales, son inventados con una
semilla fija para que siempre salga lo mismo.
"""

import numpy as np
import pandas as pd

np.random.seed(0)

# Creamos 10 pacientes ficticios con sus datos clinicos
case_id = [f"TCGA-{i:02d}" for i in range(10)]
vital_status = ["Dead", "Alive", "Dead", "Alive", "Alive",
                 "Dead", "Alive", "Alive", "Dead", "Alive"]
days_to_death = [120, None, 340, None, None, 89, None, None, 600, None]
days_to_last_follow_up = [None, 800, None, 950, 1200, None, 430, 760, None, 90]
age_at_diagnosis = [55, 62, 48, 70, 58, 61, 45, 67, 52, 59]
ajcc_pathologic_stage = ["Stage II", "Stage I", "Stage III", "Stage I", "Stage II",
                          "Stage III", "Stage I", "Stage II", "Stage IV", "Stage I"]

df_clinical = pd.DataFrame({
    "case_id": case_id,
    "vital_status": vital_status,
    "days_to_death": days_to_death,
    "days_to_last_follow_up": days_to_last_follow_up,
    "age_at_diagnosis": age_at_diagnosis,
    "ajcc_pathologic_stage": ajcc_pathologic_stage,
})

# Creamos 20 genes ficticios con expresion para los 10 pacientes
# El GENE0 lo dejamos con muchos valores vacios (NaN) a proposito,
# para luego comprobar que el script de validacion lo descarta
genes = [f"GENE{i}" for i in range(20)]
data_rna = {"gene_id": genes}

for i in range(10):
    valores = np.random.lognormal(3, 1, 20)
    if i < 6:
        valores[0] = np.nan
    data_rna[f"TCGA-{i:02d}"] = valores

df_rnaseq = pd.DataFrame(data_rna)

# Guardamos los dos ficheros en docs/tests
df_clinical.to_csv("docs/tests/test_clinical.tsv", sep="\t", index=False)
df_rnaseq.to_csv("docs/tests/test_rnaseq.tsv", sep="\t", index=False)

print("Generado: docs/tests/test_clinical.tsv (10 pacientes)")
print("Generado: docs/tests/test_rnaseq.tsv (20 genes x 10 pacientes)")