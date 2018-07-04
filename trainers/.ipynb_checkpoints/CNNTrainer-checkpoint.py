import os
import scipy
import utils
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer

from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve

class CNNTrainer(BaseTrainer):
    def __init__(self, arg, G, torch_device, recon_loss, unique_th=False):
        super(CNNTrainer, self).__init__(arg, torch_device)
        self.recon_loss = recon_loss
        
        self.G = G
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=arg.beta)
            
        self.load()
        self.best_metric = 0
        self.sigmoid = nn.Sigmoid().to(self.torch_device)
        self.unique_th = unique_th
        self.th_best = 0


    def save(self, epoch):
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.optim.state_dict(),
                    "th_best" : self.th_best,
                    "best_metric": self.best_metric
                    }, self.save_path + "/models.pth.tar")
        print("Model saved %d epoch"%(epoch))


    def load(self):
        if os.path.exists(self.save_path + "/models.pth.tar") is True:
            print("Load %s File"%(self.save_path))            
            ckpoint = torch.load(self.save_path + "/models.pth.tar")                            
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            self.G.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.th_best = ckpoint["th_best"]
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d th_best:%f"%(ckpoint["model_type"], self.start_epoch, self.th_best))
        else:
            print("Load Failed, not exists file")


    def train(self, train_loader, val_loader=None):
        print("\nStart Train")

        for epoch in range(self.start_epoch, self.epoch):
            for i, (input_, target_, _) in enumerate(train_loader):    
                self.G.train()
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_ = self.G(input_)
                recon_loss = self.recon_loss(output_, target_)
                
                self.optim.zero_grad()
                recon_loss.backward()
                self.optim.step()
            
                if (i % 50) == 0:
                    self.logger.will_write("[Train] epoch:%d loss:%f"%(epoch, recon_loss))

            if val_loader is not None:            
                self.valid(epoch, val_loader)
            else:
                self.save(epoch)
        print("End Train\n")

    def _test_foward(self, input_, target_):
        input_  = input_.to(self.torch_device)
        output_ = self.G(input_)
        output_ = self.sigmoid(output_).type(torch.FloatTensor).numpy()
        target_ = target_.type(torch.FloatTensor).numpy()
        input_  = input_.type(torch.FloatTensor).numpy()
        return input_, output_, target_


    def valid(self, epoch, val_loader):
        self.G.eval()
        with torch.no_grad():
            y_true = np.array([])
            y_pred = np.array([])
            for i, (input_, target_, _) in enumerate(val_loader):
                input_, output_, target_ = self._test_foward(input_, target_)
                target_np = utils.slice_threshold(target_, 0.5)

                y_true = np.concatenate([y_true, target_np.flatten()], axis=0)
                y_pred = np.concatenate([y_pred, output_.flatten()],   axis=0)

            roc_values = np.array(roc_curve(y_true, y_pred))
            pr_values  = np.array(precision_recall_curve(y_true, y_pred))

            f1_best, th_best = -1, 0
            for precision, recall, threshold in zip(*pr_values):
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 1
                if f1 > f1_best and f1 != 1:
                    f1_best = f1
                    th_best = threshold
                    # too much spend time
                    # np.save("%s/valid_best_roc_values.npy"%(self.save_path), roc_values)
                    # np.save("%s/valid_best_pr_values.npy"%(self.save_path),  pr_values)

            
            confusions_sum = [0, 0, 0, 0]
            for i, (input_, target_, _) in enumerate(val_loader):
                _, output_, target_ = self._test_foward(input_, target_)

                target_np = utils.slice_threshold(target_, 0.5)
                output_np = utils.slice_threshold(output_, th_best)
                target_f, output_f = target_np.flatten(), output_np.flatten()
                
                # element wise sum
                confusions    =  confusion_matrix(target_f, output_f).ravel()
                confusions_sum += confusions                

            *_, total_f1, jss = utils.get_roc_pr(*confusions_sum)
            if total_f1 > self.best_metric:
                self.best_metric = total_f1
                self.th_best = th_best
                self.save(epoch)

            self.logger.write("[Val] epoch:%d th:%f f1_best:%f f1_total:%f jss:%f"%(epoch, th_best, f1_best, total_f1, jss))
                    

    def test(self, test_loader):
        print("\nStart Test")
        self.G.eval()
        with torch.no_grad():
            y_true = np.array([])
            y_pred = np.array([])
            for i, (input_, target_, _) in enumerate(test_loader):
                input_, output_, target_ = self._test_foward(input_, target_)
                target_np = utils.slice_threshold(target_, 0.5)

                y_true = np.concatenate([y_true, target_np.flatten()], axis=0)
                y_pred = np.concatenate([y_pred, output_.flatten()],   axis=0)

            roc_values = np.array(roc_curve(y_true, y_pred))
            pr_values  = np.array(precision_recall_curve(y_true, y_pred))

            np.save("%s/test_roc_values.npy"%(self.save_path), roc_values)
            np.save("%s/test_pr_values.npy"%(self.save_path),  pr_values)
            

            confusions, cnt = [0, 0, 0, 0], 0
            f1_sum = 0
            for i, (input_, target_, f_name) in enumerate(test_loader):
                input_, output_, target_  = self._test_foward(input_, target_)

                target_np = utils.slice_threshold(target_, 0.5)
                output_np = utils.slice_threshold(output_, self.th_best)
                for batch_idx in range(0, input_.shape[0]):
                    target_b = target_np[batch_idx, 0, :, :]
                    output_b = output_np[batch_idx, 0, :, :]
                    target_f, output_f = target_b.flatten(), output_b.flatten()

                    save_path = "%s/%s"%(self.save_path, f_name[batch_idx][:-4])
                    input_norm = input_[batch_idx, 0, :, :]
                    input_norm = (input_norm - input_norm.min()) / (input_norm.max() - input_norm.min())
                    utils.image_save(save_path, input_norm, target_b, output_b)

                    confusion = confusion_matrix(target_f, output_f).ravel()
                    confusions += confusion
                    scores = utils.get_roc_pr(*confusion)
                    self.logger.will_write("[Save] fname:%s sen:%f spec:%f prec:%f rec:%f f1:%f jss:%f"%(f_name[batch_idx][:-4], *scores))

                    f1_sum += scores[-2] # image per f1
                    cnt += 1

            scores = utils.get_roc_pr(*confusions)
        self.logger.write("Best Threshold:%f sen:%f spec:%f prec:%f rec:%f f1:%f jss:%f dice:%f"%(self.th_best, *scores, f1_sum / float(cnt)))
        print("End Test\n")

    def inference(self, train_loader, valid_loader, test_loader):
        raise NotImplementedError()

        print("Start Infernce")
        os.mkdir("%s/infer"%(self.save_path))
        os.mkdir("%s/infer/Train"%(self.save_path)); os.mkdir("%s/infer/Valid"%(self.save_path)); os.mkdir("%s/infer/Test"%(self.save_path))
        loaders=[("Train", train_loader), ("Valid", valid_loader), ("Test", test_loader)]
        self.G.eval()
        with torch.no_grad():
            for path, loader in loaders:
                metric_avg, dice_avg = 0.0, 0.0
                for i, (input_, target_, f_name) in enumerate(loader):
                    input_, output_, target_ = self._test_foward(input_, target_)

                    input_np  = input_.type(torch.FloatTensor).numpy()[0, self.z_idx, :, :]
                    target_np = utils.slice_threshold(target_[0, 0, :, :], 0.5)
                    output_np = utils.slice_threshold(output_[0, 0, :, :], self.threshold)

                    jss  = self.metric(output_np, target_np)
                    dice = utils.dice(output_np, target_np)

                    if dice != 1.0:
                        save_path = "%s/infer/%s/%s"%(self.save_path, path, f_name[0][:-4])
                        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())
                        utils.image_save(save_path, input_np, target_np, output_np)

                    metric_avg += jss
                    dice_avg   += dice
