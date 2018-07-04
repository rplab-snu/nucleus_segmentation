import os
import argparse
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from sklearn.metrics import jaccard_similarity_score, f1_score

import utils
import preprocess
from NucleusLoader import NucleusLoader
from models.Fusionnet import Fusionnet
from models.unet import Unet2D
from models.unet_nonlocal import Unet_NonLocal2D
from models.unet_grid_attention import Unet_GridAttention2D
from models.unet_ct_multi_attention_dsv import Unet_CT_multi_attention_dsv_2D
from models.unet_dilated import Unet_Dilation_2D

from trainers.CNNTrainer import CNNTrainer
from trainers.GANTrainer import GANTrainer


"""parsing and configuration"""
def arg_parse():    
    desc = "Nucleus Segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")                        
    parser.add_argument('--model', type=str, default='fusion',
                        choices=['fusion', "unet", "unet_nonlocal", "unet_gridatt", "unet_multiatt", "unet_dilated"], required=True,
                        help='The type of Models | fusion | unet | unet_nonlocal | unet_gridatt | unet_multiatt | unet_dilated |')

    parser.add_argument('--in_channel', type=int, default='1',                        
                        help='The Channel of Input')
    parser.add_argument('--out_channel', type=int, default='1',                        
                        help='The Channel of Output')
    parser.add_argument('--threshold', type=float, default=0.9,                        
                        help='The Classfication Threshold')


    parser.add_argument('--dtype', type=str, default='float',
                        choices=['float', 'half'],
                        help='The torch dtype | float | half |')

    parser.add_argument('--data', type=str, default='Balance',
                        choices=['All', 'Balance', "Only_Label"],
                        help='The dataset | All | Balance | Only_Label |')
    parser.add_argument('--sampler', type=str, default='',
                        choices=['weight', ''],
                        help='The setting sampler')
    
    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--infer', action="store_true", help='The size of batch')
    parser.add_argument('--test', action="store_true", help='The size of batch')
    
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')
    
    parser.add_argument('--ngf',   type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--feature_scale', type=int, default=4)
    parser.add_argument('--dilation',      type=int, default=2)
    

    parser.add_argument('--lrG',   type=float, default=0.0005)
    parser.add_argument('--beta',  nargs="*", type=float, default=(0.5, 0.999))
    
    return parser.parse_args()

def arg_check(arg):
    if len(arg.gpus) <= 0:
        raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")
    for chk in arg.gpus:
        if chk not in "0123456789,":
            raise argparse.ArgumentTypeError("gpus must be 0,1,2 or 2,3,4 ...")
            
    check_dict = [("cpus", arg.cpus), ("epoch", arg.epoch), ("batch", arg.batch_size), ("ngf", arg.ngf), ("lrG", arg.lrG)]
    for chk in check_dict:
        if chk[1] <= 0:
            raise argparse.ArgumentTypeError("%s <= 0"%(chk[0]))
    if arg.beta[0] <= 0 or arg.beta[1] <= 0:
        raise argparse.ArgumentTypeError("betas <= 0")

if __name__ == "__main__":
    arg = arg_parse()
    arg_check(arg)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    data_path = "/data/00_Nuclues_segmentation/00_data/"
    if arg.in_channel == 1:
        data_path += "/2D/New(50_Cells)"
    elif arg.in_channel > 1 and arg.out_channel == 1:
        data_path += "/25D/%dchannel"%(arg.in_channel)

    train_path = data_path + "/%s/Train/"%(arg.data)
    valid_path = data_path + "/%s/Val/"%(arg.data)
    test_path  = data_path + "/All/Test/"

    preprocess = preprocess.get_preprocess(arg)

    train_loader = NucleusLoader(train_path, arg.batch_size, transform=preprocess, sampler=arg.sampler,
                                 channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                 shuffle=True, drop_last=True)
    valid_loader = NucleusLoader(valid_path, arg.batch_size, transform=preprocess, sampler=arg.sampler,
                                 channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                 shuffle=False, drop_last=True)
    test_loader  = NucleusLoader(test_path , 1,
                                 channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                 shuffle=False, drop_last=False)

    if arg.model == "fusion":
        net = Fusionnet(arg.in_channel, arg.out_channel, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet2D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_nonlocal":
        net = Unet_NonLocal2D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_gridatt":
        net = Unet_GridAttention2D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_multiatt":
        net = Unet_CT_multi_attention_dsv_2D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_dilated":
        net = Unet_Dilation_2D(feature_scale=arg.feature_scale, dilation=arg.dilation)
    else:
        raise NotImplementedError("Not Implemented Model")

    net = nn.DataParallel(net).to(torch_device)
    model = CNNTrainer(arg, net, torch_device, recon_loss=nn.BCEWithLogitsLoss(), metric=jaccard_similarity_score)
    if arg.infer:
        train_loader = NucleusLoader(train_path, 1, transform=preprocess,
                                     channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                     shuffle=False, drop_last=False)
        valid_loader = NucleusLoader(valid_path, 1, transform=preprocess,
                                     channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                     shuffle=False, drop_last=False)
        test_loader  = NucleusLoader(test_path , 1,
                                     channel=arg.in_channel, torch_type=arg.dtype, cpus=arg.cpus,
                                     shuffle=False, drop_last=False)

        model.inference(train_loader, valid_loader, test_loader)
    else:
        if arg.test is False:
            model.train(train_loader, valid_loader)
        model.test(test_loader)   
    utils.slack_alarm("zsef123", "Model %s Done"%(arg.save_dir))
