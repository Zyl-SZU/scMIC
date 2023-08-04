import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Module, Parameter


class AE_encoder(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.z_layer = Linear(ae_n_enc_2, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(nn.Module):
    def __init__(self, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.x_bar_layer = Linear(ae_n_dec_2, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        x_hat = self.x_bar_layer(z)
        return x_hat


class AE(nn.Module):
    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_dec_1, ae_n_dec_2, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            n_input=n_input,
            n_z=n_z)


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)

        return output


class IGAE_encoder(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, n_input, n_z, dropout):
        super(IGAE_encoder, self).__init__()
        self.gnn_1 = GNNLayer(n_input, gae_n_enc_1)
        self.gnn_2 = GNNLayer(gae_n_enc_1, gae_n_enc_2)
        self.gnn_3 = GNNLayer(gae_n_enc_2, n_z)
        self.dropout = nn.Dropout(dropout)
        self.s = nn.Sigmoid()

    def forward(self, x, adj):
        z = self.gnn_1(x, adj, active=True)
        z = self.dropout(z)
        z = self.gnn_2(z, adj, active=True)
        z = self.dropout(z)
        z_igae = self.gnn_3(z, adj, active=False)
        z_igae_adj = self.s(torch.mm(z_igae, z_igae.t()))
        return z_igae, z_igae_adj


class IGAE_decoder(nn.Module):
    def __init__(self, gae_n_dec_1, gae_n_dec_2, n_input, n_z):
        super(IGAE_decoder, self).__init__()
        self.gnn_4 = GNNLayer(n_z, gae_n_dec_1)
        self.gnn_5 = GNNLayer(gae_n_dec_1, gae_n_dec_2)
        self.gnn_6 = GNNLayer(gae_n_dec_2, n_input)
        self.s = nn.Sigmoid()

    def forward(self, z_igae, adj):
        z = self.gnn_4(z_igae, adj, active=True)
        z = self.gnn_5(z, adj, active=True)
        z_hat = self.gnn_6(z, adj, active=True)
        z_hat_adj = self.s(torch.mm(z_hat, z_hat.t()))
        return z_hat, z_hat_adj


class IGAE(nn.Module):
    def __init__(self, gae_n_enc_1, gae_n_enc_2, gae_n_dec_1, gae_n_dec_2, n_input, n_z, dropout):
        super(IGAE, self).__init__()
        self.encoder = IGAE_encoder(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            n_input=n_input,
            n_z=n_z,
            dropout=dropout)

        self.decoder = IGAE_decoder(
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            n_input=n_input,
            n_z=n_z)


class q_distribution(nn.Module):
    def __init__(self, centers):
        super(q_distribution, self).__init__()
        self.cluster_centers = centers

    def forward(self, z, z_ae, z_igae):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q_ae = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_ae = (q_ae.t() / torch.sum(q_ae, 1)).t()

        q_igae = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_centers, 2), 2))
        q_igae = (q_igae.t() / torch.sum(q_igae, 1)).t()

        return [q, q_ae, q_igae]
