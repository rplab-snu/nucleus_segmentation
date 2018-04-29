class BaseGenerator:
    def __init__(self, arg):
        self.model_type = arg.model

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        self.save_path = util.get_save_dir(arg)
        
        self.best_mse = 1

    def save(self):
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
            torch.save({"model_type"  : self.model_name,
                        "start_epoch" : self.start_epoch,
                        "network"     : self.G.state_dict(),
                        "optimizer"   : self.opim.state_dict()
                        }, os.path.join(self.save_path) + "models.pth.tar")

    def _load(self):
        pass
    
    def train(self):
        pass

    def valid(self):
        pass

    def infer(self):
        pass

    def test(self):        
        pass