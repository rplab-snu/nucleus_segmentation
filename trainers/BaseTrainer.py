import utils

class BaseTrainer:
    def __init__(self, arg):
        self.model_type = arg.model

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        
        self.output = arg.output

        self.save_path = utils.get_save_dir(arg)
    
    def save(self):
        raise NotImplementedError("notimplemented save method")

    def load(self):
        raise NotImplementedError("notimplemented save method")

    def train(self):
        raise NotImplementedError("notimplemented save method")

    def valid(self):
        raise NotImplementedError("notimplemented valid method")

    def test(self):
        raise NotImplementedError("notimplemented test method")

    def inference(self):
        raise NotImplementedError("notimplemented interence method")
        
        