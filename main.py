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
from models.UnetSH import UnetSH2D
from models.UnetRes import UnetRes2D
from models.ExFuse import ExFuse
from models.Resnet import resnet101
from models.UnetExFuse import UnetGCN, UnetGCNECRE, UnetGCNSEB, UnetExFuse

from trainers.CNNTrainer import CNNTrainer

from loss import FocalLoss, TverskyLoss

"""parsing and configuration"""


def arg_parse():
    desc = "Nucleus Segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='unet',
                        choices=['fusion', "unet", "unet_sh", "unetres", "exfuse", "unetgcn", "unetgcnseb", "unetgcnecre", "unetexfuse"], required=True)
    # Unet params
    parser.add_argument('--feature_scale', type=int, default=4)
    parser.add_argument('--sh_size', type=int, default=1)
    parser.add_argument('--pool', action="store_true", help='The size of batch')

    # FusionNet Parameters
    parser.add_argument('--ngf',   type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='BCE',
                        choices=['BCE', "tversky", "MSE"])
    # Loss Params
    parser.add_argument('--focal_gamma', type=float, default='2', help='')
    parser.add_argument('--t_alpha', type=float, default='0.3', help='')

    parser.add_argument('--dtype', type=str, default='float',
                        choices=['float', 'half'],
                        help='The torch dtype | float | half |')

    parser.add_argument('--fold', type=str, default='')

    parser.add_argument('--sampler', type=str, default='',
                        choices=['weight', ''],
                        help='The setting sampler')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--test', action="store_true", help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
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
            raise argparse.ArgumentTypeError("%s <= 0" % (chk[0]))
    if arg.beta[0] <= 0 or arg.beta[1] <= 0:
        raise argparse.ArgumentTypeError("betas <= 0")


if __name__ == "__main__":
    arg = arg_parse()
    arg_check(arg)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")

    # data_path = "/data/00_Nuclues_segmentation/DW" + arg.fold

    train_path = "dataset/dataset%s/Train/" % (arg.fold)
    valid_path = "dataset/dataset%s/Val/" % (arg.fold)
    # test_path  = data_path + "/2D/Test_FL/"
    test_path = "dataset/Test/"
    """
    train_path = "/data/00_Nuclues_segmentation/00_data/2D/New(50_Cells)/Only_Label/Train"
    valid_path = "/data/00_Nuclues_segmentation/00_data/2D/New(50_Cells)/Only_Label/Val"
    test_path = "/data/00_Nuclues_segmentation/00_data/2D/Test_FL"
    # test_path = "/home/joy/project/nuclear/dataset/test"
    """

    preprocess = preprocess.get_preprocess(arg.augment)

    train_loader = NucleusLoader(train_path, arg.batch_size, transform=preprocess, sampler=arg.sampler,
                                 torch_type=arg.dtype, cpus=arg.cpus,
                                 shuffle=True, drop_last=True)
    valid_loader = NucleusLoader(valid_path, arg.batch_size, transform=preprocess, sampler=arg.sampler,
                                 torch_type=arg.dtype, cpus=arg.cpus,
                                 shuffle=False, drop_last=False)
    test_loader = NucleusLoader(test_path, 1,
                                torch_type=arg.dtype, cpus=arg.cpus,
                                shuffle=False, drop_last=False)

    if arg.model == "fusion":
        net = Fusionnet(1, 1, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet2D(feature_scale=arg.feature_scale, is_pool=arg.pool)
    elif arg.model == "unet_sh":
        net = UnetSH2D(arg.sh_size, feature_scale=arg.feature_scale, is_pool=arg.pool)
    elif arg.model == "unetres":
        net = UnetRes2D(1, nn.InstanceNorm2d, is_pool=arg.pool)
    elif arg.model == "unetgcn":
        net = UnetGCN(arg.feature_scale, nn.InstanceNorm2d, is_pool=arg.pool)
    elif arg.model == "unetgcnseb":
        net = UnetGCNSEB(arg.feature_scale, nn.InstanceNorm2d, is_pool=arg.pool)
    elif arg.model == "unetgcnecre":
        net = UnetGCNECRE(arg.feature_scale, nn.InstanceNorm2d, is_pool=arg.pool)
    elif arg.model == "unetexfuse":
        net = UnetExFuse(arg.feature_scale, nn.InstanceNorm2d, is_pool=arg.pool)
    elif arg.model == "exfuse":
        resnet = resnet101(pretrained=True)
        net = ExFuse(resnet)
    else:
        raise NotImplementedError("Not Implemented Model")

    net = nn.DataParallel(net).to(torch_device)
    if arg.loss == "BCE":
        recon_loss = nn.BCEWithLogitsLoss()
    elif arg.loss == "tversky":
        recon_loss = TverskyLoss(arg.t_alpha, torch_device)
    elif arg.loss == "MSE":
        recon_loss = nn.MSELoss()

    model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss)
    if arg.test is False:
        # model.pre_train(train_loader, valid_loader)
        model.train(train_loader, valid_loader)
    model.test(test_loader)
    # utils.slack_alarm("zsef123", "Model %s Done"%(arg.save_dir))
