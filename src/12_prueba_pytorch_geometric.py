import torch
from torch_geometric.data import Data

# Creamos un grafo minimo de 3 nodos y 2 aristas, solo para probar
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)

grafo = Data(x=x, edge_index=edge_index)

print("Grafo creado correctamente")
print(grafo)
print("Numero de nodos:", grafo.num_nodes)
print("Numero de aristas:", grafo.num_edges)