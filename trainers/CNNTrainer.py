import os
import utils
import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer

class CNNTrainer(BaseTrainer):
    def __init__(self, arg, G, recon_loss, mse_loss=None):
        super(CNNTrainer, self).__init__(arg)
        self.output = arg.output
        self.recon_loss = recon_loss
        self.mse_loss = mse_loss     
        
        self.G = G
        if len(arg.gpus) > 1:
            self.G = nn.DataParallel(self.G).cuda()
            
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=(arg.beta1, arg.beta2))

        self.load()
        self.best_mse = 1

    def save(self):
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        torch.save({"model_type" : self.model_name,
                    "start_epoch" : self.start_epoch,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.opim.state_dict()
                    }, self.save_path + "models.pth.tar")

    def load(self):
        if os.path.exists(self.save_path + "models.pth.tar") is True:
            print("Load %s File"%(self.save_path))
            checkpoint = torch.load(load_path)
            if checkpoint['model_type'] not in ["fusion", "unet"]:
                raise Exception("model_type is %s"%(checkpoint["model_type"]))
                
            self.G.load_state_dict(checkpoint['network'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['start_epoch']


    def train(self, train_loader, val_loader=None):
        print("\nStart Train")
        best_loss = None
        # TODO : save ??
        for epoch in range(self.start_epoch, self.epoch):
            self.G.train()
            for i, (input_, target_, _) in enumerate(train_loader):
                input_  = Variable(input_).cuda()
                target_ = Variable(target_).cuda()
                output_ = G(input_)

                recon_loss = self.recon_loss(output_, target_)
                if self.mse_loss is not None:
                    mse_loss = self.mse_loss(output, target)
                    recon_loss = 0.5 * recon_loss + mse_loss
                
                self.optim.zero_grad()
                recon_loss.backward()
                self.optim.step()
            
                if (i % 100) == 0:
                    print("[Train] epoch[%d/%d:%d] loss:%f"%(epoch, self.epoch, i, recon_loss.data[0]))

            if val_loader is not None:            
                self.valid(epoch, val_loader)
            else:
                self.save()
        print("End Train\n")


    def valid(self, epoch, val_loader):
        print("\nStart Val")
        self.G.eval()
        mse_loss = 0
        for i, (input_, target_, _) in enumerate(val_loader):
            input_  = Variable(input_ , volatile=True).cuda()
            target_ = Variable(target_, volatile=True).cuda()
            output_ = G(input_)

            target_ = target_.cpu.data[0].numpy()
            output_ = output_.cpu.data[0].numpy()

            mse_loss += utils.mse(output_, target_)
            
        mse_loss /= len(val_loader)
        print("[Val] epoch:{} / best_mse:{}\n".format(epoch, best_mse))
        if mse_loss < self.best_mse:
            self.best_mse = mse_loss.data[0]
            self.save()
        print("End Val\n")
        
           
    def test(self, test_loader):
        print("\nStart Test")

        self.G.eval()
        for i, (input_, target_, f_name) in enumerate(test_loader):
            input_  = Variable(input_, volatile=True).cuda()
            output_ = self.G(input_)

            input_  = utils.tr_to_np(input_)
            output_ = utils.tr_to_np(output_)
            target_ = utils.tr_to_np(target_)

            recon_loss = self.recon_loss(output_var, target_var)
            print("[Test] cnt:{} loss:{}".format(i, recon_loss.data[0]))
            save_path = os.path.join(self.save_dir, "_%d"%(i))
            utils.image_save(save_path, input_, output__, target_)
        print("End Test\n")

    def inference(self, infer_loader):
        print("\nStart Inference")
        self.G.eval()
        for i, (input_, f_name) in enumerate(infer_loader):
            input_  = Variable(input_).cuda()
            output_ = self.G(input_)

            input_  = utils.tr_to_np(input_)
            output_ = utils.tr_to_np(output_)

            recon_loss = self.recon_loss(output_var, target_var)
            print("[Infer] cnt:{} loss:{}".format(i, recon_loss.data[0]))
            save_path = os.path.join(self.save_dir, "_%d"%(i))
            utils.image_save(save_path, input_, output_)
        print("End Inference\n")

