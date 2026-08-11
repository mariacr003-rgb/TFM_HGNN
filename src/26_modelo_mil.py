import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# Bloque 8: MIL (Multiple Instance Learning) sobre WSI, formulas de la
# Seccion 4.5 del TFM (Ilse et al. 2018).
#
# NOTA IMPORTANTE sobre el encoder ResNet-50: la formula especifica un
# "encoder ResNet-50 pre-entrenado sobre datos histologicos de TCGA".
# No se dispone de un checkpoint asi en este proyecto (no publicado de
# forma directamente reutilizable sin mas investigacion); se usa el
# ResNet-50 preentrenado en ImageNet de torchvision como sustituto
# practico para las pruebas de este bloque, documentado aqui como
# desviacion explicita de la formula, no un intento de igualarla. Si
# se decide seguir adelante con MIL, buscar un checkpoint
# histologia-especifico (ej. CTransPath, RetCCL, Ciga et al. SimCLR)
# queda como tarea pendiente.
#
# EXCEPCION AL ESTILO DEL PROYECTO ("sin clases"): igual que
# ModeloGATGCN y ModeloVGAE, PyTorch exige subclases de nn.Module.


class EncoderResNetMIL(nn.Module):
    # Envuelve ResNet-50 de torchvision, quitando la capa final de
    # clasificacion (1000 clases de ImageNet) para usar el vector de
    # 2048 dimensiones de la penultima capa (global average pool) como
    # embedding h_k de cada parche.
    def __init__(self, congelar_pesos=True):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.dim_salida = resnet.fc.in_features  # 2048
        self.tronco = nn.Sequential(*list(resnet.children())[:-1])  # todo menos fc
        if congelar_pesos:
            # Sin GPU y con recursos limitados, no se plantea
            # fine-tuning del ResNet-50 en este bloque (coste
            # prohibitivo, ver estimaciones del plan de diseño):
            # actua solo como extractor de caracteristicas fijo.
            for parametro in self.tronco.parameters():
                parametro.requires_grad = False

    def forward(self, parches):
        # parches: [n_parches, 3, 256, 256], ya normalizados (ver
        # 27_procesar_wsi_paciente.py). Devuelve [n_parches, 2048].
        h = self.tronco(parches)  # [n_parches, 2048, 1, 1]
        return h.flatten(1)


class AtencionMIL(nn.Module):
    # Mecanismo de atencion de Ilse et al. 2018:
    #   a_k = softmax_k( w^T * tanh(V * h_k^T) )
    #   z = suma_k (a_k * h_k)
    def __init__(self, dim_entrada=2048, dim_oculto=128):
        super().__init__()
        self.V = nn.Linear(dim_entrada, dim_oculto, bias=False)
        self.w = nn.Linear(dim_oculto, 1, bias=False)

    def forward(self, h):
        # h: [n_parches, dim_entrada] - embeddings de todos los
        # parches de UN paciente (una WSI, o la union de varias si el
        # paciente tiene mas de una lamina).
        scores = self.w(torch.tanh(self.V(h)))  # [n_parches, 1]
        a = torch.softmax(scores, dim=0)  # [n_parches, 1]
        z = (a * h).sum(dim=0)  # [dim_entrada]
        return z, a.squeeze(-1)
