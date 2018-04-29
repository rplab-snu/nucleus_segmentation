import os
import argparse
import torch.nn as nn
from MARLoader import MARLoader
from models.Fusionnet import Fusionnet
from models.Unet import Unet
from trainers.CNNTrainer import CNNTrainer
from trainers.GANTrainer import GANTrainer


"""parsing and configuration"""
def arg_parse():    
    desc = "Metal Artifact Reduction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0",
                        help="Select GPU Numbering | 0,1,2,3,4 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")                        
    parser.add_argument('--model', type=str, default='fusion',
                        choices=['fusion', "pix2pix"], required=True,
                        help='The type of Models | fusion | pix2pix |')
    parser.add_argument('--output', type=str, default='sin2res',
                        choices=['sin2res', 'img2res', 'img2img'], required=True,
                        help='The name of outputs | sin2res | img2res | img2img |')
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--infer', action="store_true", help='The size of batch')
    
    parser.add_argument('--save_dir', type=str, default='outs',
                        help='Directory name to save the model')
    
    parser.add_argument('--ngf',   type=int, default=56)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--lrG',   type=float, default=0.0002)
    parser.add_argument('--lrD',   type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    return parser.parse_args()

def arg_check(arg):
    if len(arg.gpus) <= 0:
        raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")
    for chk in arg.gpus:
        if chk not in "0123456789,":
            raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")
            
    check_dict = [("cpus", arg.cpus), ("epoch", arg.epoch), ("batch", arg.batch_size), ("ngf", arg.ngf)]
    for chk in check_dict:
        if chk[1] <= 0:
            raise argparse.ArgumentTypeError("%s <= 0"%(chk[0]))
    if arg.lrG <= 0 or arg.lrD <= 0:
        raise argparse.ArgumentTypeError("lr <= 0")
    if arg.beta1 <= 0 or arg.beta2 <= 0:
        raise argparse.ArgumentTypeError("betas <= 0")

if __name__ == "__main__":    
    arg = arg_parse()
    arg_check(arg)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    if os.path.exists(arg.save_dir) is False:
        os.mkdir(arg.save_dir)

    output_type = arg.output
    train_path = "../00_Data/npy/Train_sinogram"
    valid_path = "../00_Data/npy/Val"
    # test_path  = ""

    valid_loader = MARLoader(valid_path, arg.batch_size, image_type=arg.output, cpus=arg.cpus,)
    train_loader = MARLoader(train_path, arg.batch_size, image_type=arg.output, cpus=arg.cpus,)

    if arg.model == "fusion":
        fusionnet = Fusionnet(1, 1, arg.ngf, arg.output, arg.clamp).cuda()
        model = CNNTrainer(arg, fusionnet, recon_loss=nn.L1Loss(), mse_loss=nn.MSELoss())
    elif arg.model == "unet":
        # for example
        # unet = Unet(1, 1, arg.ngf, arg.output, arg.clamp).cuda()
        # model = CNNTrainer(arg, Unet, recon_loss=nn.L1Loss(), mse_loss=nn.MSELoss())
        raise NotImplementedError("Not Implemented Unet")

    elif arg.model == "pix2pix":
        # G = ...Net
        # D = ...Discriminator
        # model = GANTrainer(...)
        raise NotImplementedError("Not Implemented Model")
        
    else:
        raise NotImplementedError("Not Implemented Model")

    if arg.infer:
        model.inference(test_loader)
    else:
        model.train(train_loader, valid_loader)
        # model.test(test_loader)
    
