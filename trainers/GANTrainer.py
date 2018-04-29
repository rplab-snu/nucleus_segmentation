import BaseTrainer

# TODO : Implementation
class GANTrainer(BaseTrainer):
    def __init__(self, arg, G, D, recon_loss, mse_loss):
        self.model = arg.model_type

        self.output = arg.output
        self.epoch = epoch
        self.batch_size = arg.batch_size
        self.G = G
        self.D = D
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=(arg.beta1, arg.beta2))
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr=arg.lrD, betas=(arg.beta1, arg.beta2))
        
        self.recon_loss = recon_loss
        self.mse_loss = mse_loss

        self.best_mse = 1


    def save(self):
        pass

    def load(self):
        pass

    def train(self, train_loader, val_loader=None):
        pass

    def valid(self, epoch, val_loader):
        pass

    def test(sef, test_loader):
        pass