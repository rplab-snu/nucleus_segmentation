import utils
from Logger import Logger

class BaseTrainer:
    def __init__(self, arg, torch_device):
        self.torch_device = torch_device 
        
        self.model_type = arg.model

        self.z_idx = 0

        self.epoch = arg.epoch
        self.start_epoch = 0

        self.batch_size = arg.batch_size
        
        self.save_path = utils.get_save_dir(arg)
        self.log_file_path = self.save_path+"/log.txt"

        self.logger = Logger(arg, self.save_path)
    
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
        
        
