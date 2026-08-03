import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, GCNConv

# Bloque 6: definicion del modelo GAT (genes) + GCN (pacientes), sin la
# capa de agregacion por vias (Capa 2, fuera de alcance de este bloque:
# ver runlog del Bloque 6 para la discusion completa).
#
# EXCEPCION AL ESTILO DEL PROYECTO ("sin clases, funciones basicas"):
# PyTorch exige subclases de nn.Module para registrar los pesos
# entrenables que el optimizador debe actualizar; no hay alternativa
# idiomatica sin clases usando GATConv/GCNConv de PyTorch Geometric. Es
# la unica excepcion del Bloque 6 (documentada en el runlog): los demas
# scripts nuevos (19, 20, 22) se mantienen sin clases.
#
# Arquitectura (formulas 1-3 de la Seccion 4 del TFM):
#   1) GAT sobre el grafo gen-gen (proyeccion del PPI de STRING, ver
#      19_proyectar_grafo_gen_gen.py): 2 capas, K=8 cabezas de atencion,
#      sigma=ELU (formulas 1 y 2, Tanvir et al. 2024). Los mismos pesos
#      se aplican a cada paciente (mismo edge_index, distinto x por
#      paciente).
#   2) Pooling medio sobre los embeddings de gen de cada paciente para
#      obtener un vector fijo por paciente. Sustituye a "embeddings de
#      vias" de la Capa 2 (no implementada en este bloque): decision
#      razonada y documentada en el runlog del Bloque 6, no una
#      reformulacion de la formula 3 del TFM.
#   3) Grafo paciente-paciente por k-NN (k=20 por defecto, similitud
#      coseno, simetrizado) sobre ese vector, recalculado en cada
#      forward pass (dinamico: el coste de k-NN sobre ~200 puntos es
#      insignificante frente al coste del GAT; ver justificacion en el
#      runlog).
#   4) GCN sobre el grafo paciente-paciente (formula 3, Kesimoglu y
#      Bozdag 2023): 2 capas, usando GCNConv de PyTorch Geometric, que
#      ya implementa D~^(-1/2) A~ D~^(-1/2) H W con A~=A+I.
#   5) Capa lineal final -> log-riesgo escalar por paciente (entrada de
#      la perdida de Cox, implementada en 22_entrenar_gat_gcn.py).


class CodificadorGenGAT(nn.Module):
    # GAT sobre el grafo gen-gen (formulas 1-2): 2 capas, K cabezas de
    # atencion concatenadas en la primera capa, promediadas en la
    # segunda para fijar la dimension de salida por gen.
    def __init__(self, dim_entrada=3, dim_oculto=16, dim_salida=32, cabezas=8):
        super().__init__()
        self.capa1 = GATConv(dim_entrada, dim_oculto, heads=cabezas, concat=True)
        self.capa2 = GATConv(dim_oculto * cabezas, dim_salida, heads=cabezas, concat=False)
        self.activacion = nn.ELU()

    def forward(self, x, edge_index):
        h = self.activacion(self.capa1(x, edge_index))
        h = self.activacion(self.capa2(h, edge_index))
        return h  # [n_genes, dim_salida]


def construir_grafo_knn_pacientes(embeddings_paciente, k):
    # k-NN dinamico sobre el vector agregado por paciente (ver punto 3
    # de la cabecera del fichero): similitud coseno, simetrizado (union
    # de aristas i->j / j->i). El top-k no es diferenciable, por eso
    # ModeloGATGCN llama a esta funcion siempre con
    # embeddings_paciente.detach(): la topologia no propaga gradiente,
    # pero los VALORES que luego entran a la GCN si lo hacen.
    n_pacientes = embeddings_paciente.shape[0]
    k_efectivo = min(k, n_pacientes - 1)
    emb_norm = nn.functional.normalize(embeddings_paciente, dim=1)
    similitud = emb_norm @ emb_norm.T
    similitud.fill_diagonal_(-float("inf"))
    _valores, vecinos = similitud.topk(k_efectivo, dim=1)

    origen = torch.arange(n_pacientes).unsqueeze(1).expand(-1, k_efectivo).reshape(-1)
    destino = vecinos.reshape(-1)
    origen_sim = torch.cat([origen, destino])
    destino_sim = torch.cat([destino, origen])
    edge_index = torch.stack([origen_sim, destino_sim], dim=0)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


class PropagadorPacienteGCN(nn.Module):
    # GCN sobre el grafo paciente-paciente (formula 3): 2 capas GCNConv
    # (normalizacion simetrica D~^(-1/2) A~ D~^(-1/2) ya incluida).
    def __init__(self, dim_entrada=32, dim_oculto=32, dim_salida=32):
        super().__init__()
        self.capa1 = GCNConv(dim_entrada, dim_oculto, add_self_loops=True, normalize=True)
        self.capa2 = GCNConv(dim_oculto, dim_salida, add_self_loops=True, normalize=True)
        self.activacion = nn.ELU()

    def forward(self, h, edge_index):
        h = self.activacion(self.capa1(h, edge_index))
        h = self.activacion(self.capa2(h, edge_index))
        return h


class ModeloGATGCN(nn.Module):
    def __init__(self, dim_entrada_gen=3, dim_oculto_gat=16, dim_salida_gat=32,
                 cabezas_gat=8, dim_oculto_gcn=32, dim_salida_gcn=32, k_vecinos=20):
        super().__init__()
        self.codificador_gen = CodificadorGenGAT(dim_entrada_gen, dim_oculto_gat, dim_salida_gat, cabezas_gat)
        self.propagador_paciente = PropagadorPacienteGCN(dim_salida_gat, dim_oculto_gcn, dim_salida_gcn)
        self.cabeza_riesgo = nn.Linear(dim_salida_gcn, 1)
        self.k_vecinos = k_vecinos

    def forward(self, x_pacientes, edge_index_gen_gen):
        # x_pacientes: [n_pacientes, n_genes, 3], atributos ya
        # normalizados (ver 20_preprocesar_atributos_gen.py). El GAT se
        # aplica paciente a paciente (mismo edge_index, mismos pesos,
        # distinto x): PyG no permite compartir topologia entre grafos
        # de un batch sin replicarla, y aqui se prioriza la version mas
        # simple mientras se mide el coste real (ver smoke test en el
        # runlog); si resulta ser el cuello de botella, se sustituira
        # por un batching vectorizado con torch_geometric.data.Batch.
        #
        # NOTA MEMORIA (smoke test OOM, ver runlog Bloque 6): sin
        # checkpointing, el grafo de autograd del GAT (~20k nodos,
        # ~457k aristas; solo el gather interno de GATConv de "x_j" por
        # arista ya son cientos de MB por paciente y capa) se retiene
        # ENTERO para cada paciente del pliegue de entrenamiento hasta
        # el backward() unico al final del epoch (ver
        # entrenar_un_pliegue en 22_entrenar_gat_gcn.py), multiplicando
        # el pico de memoria por el numero de pacientes del pliegue.
        # torch.utils.checkpoint recalcula el forward del GAT de cada
        # paciente durante el backward en vez de guardar sus
        # activaciones intermedias, acotando el pico a UN paciente en
        # vez de N. use_reentrant=False (API recomendada actual): no
        # requiere que las entradas tengan requires_grad=True, basta con
        # que el modulo tenga parametros entrenables.
        embeddings = [
            checkpoint(self.codificador_gen, x_pacientes[p], edge_index_gen_gen, use_reentrant=False).mean(dim=0)
            for p in range(x_pacientes.shape[0])
        ]
        h0_pacientes = torch.stack(embeddings, dim=0)  # [n_pacientes, dim_salida_gat]

        edge_index_pacientes = construir_grafo_knn_pacientes(h0_pacientes.detach(), self.k_vecinos)
        h_final = self.propagador_paciente(h0_pacientes, edge_index_pacientes)
        riesgo = self.cabeza_riesgo(h_final).squeeze(-1)  # [n_pacientes]
        return riesgo
