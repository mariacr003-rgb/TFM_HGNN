import torch
from torch import nn
from torch_geometric.nn import GCNConv

# Bloque 7: VGAE (Variational Graph Autoencoder) para imputacion
# generativa de datos faltantes (formulas de la Seccion 4.4 del TFM,
# Kipf y Welling 2016).
#
# A diferencia del GAT+GCN (Bloque 6, 21_modelo_gat_gcn.py), que
# aplica el GAT sobre el grafo GEN-GEN (PPI de STRING) paciente a
# paciente, este VGAE opera sobre el grafo PACIENTE-PACIENTE (PSN,
# k-NN por similitud coseno): la formula del encoder produce, "para
# cada paciente i", una media mu_i y una varianza sigma_i^2 - los
# nodos del grafo son pacientes, no genes.
#
# Independencia del GAT+GCN (Bloque 6): el texto especifica que el
# VGAE "se pre-entrena de forma independiente... antes del
# entrenamiento conjunto" (Kipf y Welling, 2016). Por eso el grafo de
# pacientes de ESTE modulo no se construye a partir de embeddings del
# GAT entrenado (seria una dependencia circular: el GAT necesita
# atributos ya imputados para correr), sino de una representacion
# directa de los datos ya imputados por media (ver
# 24_entrenar_vgae.py:construir_grafo_pacientes). Se reutiliza la
# FUNCION de construccion de grafo k-NN de 21_modelo_gat_gcn.py
# (construir_grafo_knn_pacientes), no el grafo en si.
#
# EXCEPCION AL ESTILO DEL PROYECTO ("sin clases"): igual que
# ModeloGATGCN, PyTorch exige subclases de nn.Module para registrar
# pesos entrenables.
#
# Arquitectura (Kipf y Welling 2016 - tronco GCN compartido + dos
# cabezas GCN separadas para mu y log-varianza; decision confirmada
# con el usuario: no dos pilas GCN totalmente independientes):
#   1) Encoder: la entrada por paciente concatena el vector de
#      atributos por gen ya imputado/normalizado (ver
#      20_preprocesar_atributos_gen.py) CON la mascara de validez
#      (para que la red sepa que celdas son dato real y cuales son
#      placeholder de imputacion). Una capa GCN compartida -> ELU ->
#      dos capas GCN finales separadas (mu, log-varianza).
#   2) Reparametrizacion: z = mu + eps*sigma, eps~N(0,I). En
#      evaluacion (model.eval()) se usa z=mu directamente (sin ruido),
#      practica estandar de VAE.
#   3) Decoder: TAMBIEN basado en GCN sobre el mismo grafo de
#      pacientes (decision confirmada con el usuario, no un MLP
#      plano), simetrico al encoder. Reconstruye SOLO el bloque de
#      valores (genes x canales), no la mascara auxiliar de entrada.


class EncoderVGAE(nn.Module):
    def __init__(self, dim_entrada, dim_oculto, dim_latente):
        super().__init__()
        self.capa_compartida = GCNConv(dim_entrada, dim_oculto, add_self_loops=True, normalize=True)
        self.capa_mu = GCNConv(dim_oculto, dim_latente, add_self_loops=True, normalize=True)
        self.capa_logvar = GCNConv(dim_oculto, dim_latente, add_self_loops=True, normalize=True)
        self.activacion = nn.ELU()

    def forward(self, x, edge_index):
        h = self.activacion(self.capa_compartida(x, edge_index))
        mu = self.capa_mu(h, edge_index)
        # logvar sin acotar puede desbordar exp() en float32 tras un
        # solo paso de gradiente (mas probable aqui por la dimension
        # de entrada inusualmente grande, ~120k) - visto en la practica
        # en el smoke test (epoca 1 razonable, epoca 2 diverge a NaN).
        # Acotar a [-10, 10] es la practica estandar en VAEs para
        # evitar esto: exp(10)=22026 y exp(-10)=4.5e-5, ambos seguros.
        logvar = torch.clamp(self.capa_logvar(h, edge_index), min=-10.0, max=10.0)
        return mu, logvar


class DecoderVGAE(nn.Module):
    def __init__(self, dim_latente, dim_oculto, dim_salida):
        super().__init__()
        self.capa1 = GCNConv(dim_latente, dim_oculto, add_self_loops=True, normalize=True)
        self.capa2 = GCNConv(dim_oculto, dim_salida, add_self_loops=True, normalize=True)
        self.activacion = nn.ELU()

    def forward(self, z, edge_index):
        h = self.activacion(self.capa1(z, edge_index))
        return self.capa2(h, edge_index)


class ModeloVGAE(nn.Module):
    def __init__(self, n_genes, n_canales=3, dim_oculto=128, dim_latente=32):
        super().__init__()
        self.dim_valores = n_genes * n_canales
        dim_entrada = self.dim_valores * 2  # valores + mascara concatenados
        self.encoder = EncoderVGAE(dim_entrada, dim_oculto, dim_latente)
        self.decoder = DecoderVGAE(dim_latente, dim_oculto, self.dim_valores)

    def forward(self, x_valores, x_mascara, edge_index):
        # x_valores, x_mascara: [n_pacientes, dim_valores] (ya
        # aplanados por gen*canal). La mascara que se concatena aqui
        # es la mascara ACTIVA para esta llamada (durante el
        # entrenamiento, la de entrenamiento con las celdas retenidas
        # ocultas; en la reconstruccion final, la mascara real
        # completa) - ver 24_entrenar_vgae.py.
        x_entrada = torch.cat([x_valores, x_mascara], dim=1)
        mu, logvar = self.encoder(x_entrada, edge_index)
        if self.training:
            sigma = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(sigma) * sigma
        else:
            z = mu  # sin ruido en evaluacion, practica estandar de VAE
        x_reconstruido = self.decoder(z, edge_index)
        return x_reconstruido, mu, logvar


def perdida_elbo(x_reconstruido, x_verdad, mascara, mu, logvar, beta_kl=1.0):
    # -ELBO = -E[log p(X_obs|Z)] + beta_kl * KL(q(Z|X) || p(Z)),
    # p(Z)=N(0,I) (formulas de la Seccion 4.4; beta_kl=1.0 es el ELBO
    # estandar). El termino de reconstruccion se aproxima con error
    # cuadratico medio (equivalente, salvo constantes, a -log p bajo
    # un decoder Gaussiano de varianza fija), calculado UNICAMENTE
    # sobre las celdas indicadas por 'mascara' (ver
    # 24_entrenar_vgae.py para como se construye).
    diff2 = (x_reconstruido - x_verdad) ** 2
    n_validas = mascara.sum().clamp(min=1)
    perdida_reconstruccion = (diff2 * mascara).sum() / n_validas

    # KL de una Gaussiana diagonal q(z|x)=N(mu,sigma^2) frente a
    # N(0,I): forma cerrada estandar, sumada sobre dims latentes,
    # promediada sobre pacientes.
    kl_por_paciente = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    divergencia_kl = kl_por_paciente.mean()

    # beta_kl: peso de calentamiento (KL annealing) aplicado por el
    # bucle de entrenamiento en 24_entrenar_vgae.py, NO calculado
    # aqui. Sin el (beta_kl=1.0 desde la epoca 1), el KL puede saturar
    # el clamp de logvar y dominar la perdida por varios ordenes de
    # magnitud antes de que el encoder aprenda nada util de la
    # reconstruccion - fenomeno conocido en la literatura de VAEs
    # ("KL collapse"/"posterior collapse" temprano), ver runlog del
    # Bloque 7 para el hallazgo concreto observado en el smoke test.
    perdida_total = perdida_reconstruccion + beta_kl * divergencia_kl
    return perdida_total, perdida_reconstruccion, divergencia_kl
