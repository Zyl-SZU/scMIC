import tqdm
from torch.optim import Adam

from utils import *
from encoder import *
from scMIC import scMIC
from data_loader import load_data


def pretrain_ae(model, x):
    print("Pretraining AE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        loss = F.mse_loss(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + opt.args.alpha_value * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def pre_train(model, X1, A1, X2, A2):
    print("Pretraining fusion model...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.fus_epoch)):

        # input & output
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, _, _, _, _, cons = model(X1, A1, X2, A2, pretrain=True)

        L_DRR = drr_loss(cons)
        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)
        loss = L_REC1 + L_REC2 + L_DRR
 
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), './model_pretrained/{}_pretrain.pkl'.format(opt.args.name))
    

def train(model, X1, A1, X2, A2, y):
    if not opt.args.pretrain:
        # loading pretrained model
        model.load_state_dict(torch.load('./model_pretrained/{}_pretrain.pkl'.format(opt.args.name), map_location='cpu'))

        with torch.no_grad():
            _, _, _, _, _, _, _, _, Z1, Z2, _ = model(X1, A1, X2, A2)
        
        _, _, _, _, centers1 = clustering(Z1, y)
        _, _, _, _, centers2 = clustering(Z2, y)

        # initialize cluster centers
        model.cluster_centers1.data = torch.tensor(centers1).to(opt.args.device)
        model.cluster_centers2.data = torch.tensor(centers2).to(opt.args.device)
    
    print("Training...")

    optimizer = Adam(model.parameters(), lr=(opt.args.lr))

    pbar = tqdm.tqdm(range(opt.args.epoch), ncols=200)
    for epoch in pbar:

        # input & output
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Q1, Q2, Z1, Z2, cons = model(X1, A1, X2, A2)

        L_DRR = drr_loss(cons)
        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)

        if opt.args.first_view == 'RNA':
            if epoch % 400 < 200:
                L_KL1 = distribution_loss(Q1, target_distribution(Q1[0].data))
                L_KL2 = distribution_loss(Q2, target_distribution(Q1[0].data))
            else:
                L_KL1 = distribution_loss(Q1, target_distribution(Q2[0].data))
                L_KL2 = distribution_loss(Q2, target_distribution(Q2[0].data))
        else:
            if epoch % 400 < 200:
                L_KL1 = distribution_loss(Q1, target_distribution(Q2[0].data))
                L_KL2 = distribution_loss(Q2, target_distribution(Q2[0].data))
            else:
                L_KL1 = distribution_loss(Q1, target_distribution(Q1[0].data))
                L_KL2 = distribution_loss(Q2, target_distribution(Q1[0].data))

        loss = L_REC1 + L_REC2 + L_DRR + opt.args.lambda3 * (L_KL1 + L_KL2)
 
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clustering & evaluation
        ari, nmi, ami, acc, _ = assignment((Q1[0] + Q2[0]).data, y)

        pbar.set_postfix({'loss':'{0:1.4f}'.format(loss), 'ARI':'{0:1.4f}'.format(ari),'NMI':'{0:1.4f}'.format(nmi),
                          'AMI':'{0:1.4f}'.format(ami),'ACC':'{0:1.4f}'.format(acc)})
    
        if ari > opt.args.ari:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.ami = ami
            best_epoch = epoch

    pbar.close()
    
    print("Best_epoch: {},".format(best_epoch),"ARI: {:.4f},".format(opt.args.ari), "NMI: {:.4f},".format(opt.args.nmi), 
            "AMI: {:.4f}".format(opt.args.ami), "ACC: {:.4f},".format(opt.args.acc))
    
    print("Final_epoch: {},".format(epoch),"ARI: {:.4f},".format(ari), "NMI: {:.4f},".format(nmi), 
            "AMI: {:.4f}".format(ami), "ACC: {:.4f},".format(acc))

if __name__ == '__main__':
    # setup
    print("setting:")

    setup_seed(opt.args.seed)

    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("lambda1 value : {}".format(opt.args.lambda1))
    print("lambda2 value : {}".format(opt.args.lambda2))
    print("lambda3 value : {}".format(opt.args.lambda3))
    print("alpha value   : {:.0e}".format(opt.args.alpha_value))
    print("k value       : {}".format(opt.args.k))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("------------------------------")

    # load data
    Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)
    Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)

    Xr = numpy_to_torch(Xr).to(opt.args.device)
    Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)

    Xa = numpy_to_torch(Xa).to(opt.args.device)
    Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

    ae1 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z).to(opt.args.device)

    ae2 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z).to(opt.args.device)

    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    if opt.args.pretrain:
        pretrain_ae(ae1, Xr)
        pretrain_ae(ae2, Xa)

        pretrain_gae(gae1, Xr, Ar)
        pretrain_gae(gae2, Xa, Aa)

        model = scMIC(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        pre_train(model, Xr, Ar, Xa, Aa)
    else:
        model = scMIC(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        train(model, Xr, Ar, Xa, Aa, y)
