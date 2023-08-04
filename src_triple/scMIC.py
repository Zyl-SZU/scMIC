import opt
from encoder import *


class scMIC(nn.Module):
    def __init__(self, ae1, ae2, ae3, gae1, gae2, gae3, n_node=None):
        super(scMIC, self).__init__()

        self.ae1 = ae1
        self.ae2 = ae2
        self.ae3 = ae3

        self.gae1 = gae1
        self.gae2 = gae2
        self.gae3 = gae3

        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, opt.args.n_z), 0.5), requires_grad=True)  # Z_ae, Z_igae
        self.alpha = Parameter(torch.zeros(1))   # ZG, ZL

        self.cluster_centers1 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers2 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)
        self.cluster_centers3 = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.n_z), requires_grad=True)

        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        torch.nn.init.xavier_normal_(self.cluster_centers2.data)
        torch.nn.init.xavier_normal_(self.cluster_centers3.data)

        self.q_distribution1 = q_distribution(self.cluster_centers1)
        self.q_distribution2 = q_distribution(self.cluster_centers2)
        self.q_distribution3 = q_distribution(self.cluster_centers3)

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(n_node, opt.args.n_clusters),
            nn.Softmax(dim=1)
        )

    
    def emb_fusion(self, adj, z_ae, z_igae):
        z_i = self.a * z_ae + (1 - self.a) * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.alpha * z_g + z_l

        return z_tilde

    def forward(self, x1, adj1, x2, adj2, x3, adj3, pretrain=False):
        # node embedding encoded by AE
        z_ae1 = self.ae1.encoder(x1)
        z_ae2 = self.ae2.encoder(x2)
        z_ae3 = self.ae3.encoder(x3)

        # node embedding encoded by IGAE
        z_igae1, a_igae1 = self.gae1.encoder(x1, adj1)
        z_igae2, a_igae2 = self.gae2.encoder(x2, adj2)
        z_igae3, a_igae3 = self.gae3.encoder(x3, adj3)

        z1 = self.emb_fusion(adj1, z_ae1, z_igae1)
        z2 = self.emb_fusion(adj2, z_ae2, z_igae2)
        z3 = self.emb_fusion(adj3, z_ae3, z_igae3)

        z1_tilde = self.label_contrastive_module(z1.T)
        z2_tilde = self.label_contrastive_module(z2.T)
        z3_tilde = self.label_contrastive_module(z3.T)

        cons = [z1, z2, z3, z1_tilde, z2_tilde, z3_tilde]

        # AE decoding
        x_hat1 = self.ae1.decoder(z1)
        x_hat2 = self.ae2.decoder(z2)
        x_hat3 = self.ae3.decoder(z3)

        # IGAE decoding
        z_hat1, z_adj_hat1 = self.gae1.decoder(z1, adj1)
        a_hat1 = a_igae1 + z_adj_hat1

        z_hat2, z_adj_hat2 = self.gae2.decoder(z2, adj2)
        a_hat2 = a_igae2 + z_adj_hat2

        z_hat3, z_adj_hat3 = self.gae3.decoder(z3, adj3)
        a_hat3 = a_igae3 + z_adj_hat3

        if not pretrain:
            # the soft assignment distribution Q
            Q1 = self.q_distribution1(z1, z_ae1, z_igae1)
            Q2 = self.q_distribution2(z2, z_ae2, z_igae2)
            Q3 = self.q_distribution3(z3, z_ae3, z_igae3)
        else:
            Q1, Q2, Q3 = None, None, None
           
        return x_hat1, z_hat1, a_hat1, x_hat2, z_hat2, a_hat2, x_hat3, z_hat3, a_hat3, Q1, Q2, Q3, z1, z2, z3, cons
