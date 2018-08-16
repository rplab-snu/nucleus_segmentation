import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer
from models.layers.UnetResLayer import weights_init_kaiming

from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve


class CNNTrainer(BaseTrainer):
    def __init__(self, arg, G, torch_device, recon_loss):
        super(CNNTrainer, self).__init__(arg, torch_device)
        self.recon_loss = recon_loss

        self.G = G
        self.lrG = arg.lrG
        self.beta = arg.beta
        self.fold = arg.fold
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=arg.beta)

        self.best_metric = 0
        self.sigmoid = nn.Sigmoid().to(self.torch_device)

        self.load()
        self.prev_epoch_loss = 0

    def save(self, epoch, filename="models"):
        save_path = self.save_path + "/fold%s"%(self.fold)
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        torch.save({"model_type": self.model_type,
                    "start_epoch": epoch + 1,
                    "network": self.G.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "best_metric": self.best_metric,
                    }, save_path + "/%s.pth.tar" % (filename))
        print("Model saved %d epoch" % (epoch))

    def load(self):
        save_path = self.save_path + "/fold%s"%(self.fold)
        if os.path.exists(save_path + "/models.pth.tar") is True:
            print("Load %s File" % (save_path))
            ckpoint = torch.load(save_path + "/models.pth.tar")
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.G.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d" % (ckpoint["model_type"], self.start_epoch))
        else:
            print("Load Failed, not exists file")

    def _init_model(self):
        for m in self.G.modules():
            m.apply(weights_init_kaiming)
        self.optim = torch.optim.Adam(self.G.parameters(), lr=self.lrG, betas=self.beta)

    def pre_train(self, train_loader, val_loader):
        print("PretrainStart")
        cnt, f1 = 0, 0
        while f1 < 0.85:
            # Model Init
            self._init_model()

            for epoch in range(3):
                for i, (input_, target_, _) in enumerate(train_loader):
                    self.G.train()
                    input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                    output_ = self.G(input_)
                    recon_loss = self.recon_loss(output_, target_)

                    self.optim.zero_grad()
                    recon_loss.backward()
                    self.optim.step()

            self.G.eval()
            with torch.no_grad():
                confusions_sum = [0, 0, 0, 0]
                # F1(per dataset) , JSS, Dice(per image)
                for i, (input_, target_, _) in enumerate(val_loader):
                    _, output_, target_ = self._test_foward(input_, target_)

                    target_np = utils.slice_threshold(target_, 0.5)
                    output_np = utils.slice_threshold(output_, 0.5)
                    target_f, output_f = target_np.flatten(), output_np.flatten()

                    # element wise sum
                    confusions_sum += confusion_matrix(target_f, output_f).ravel()

                f1 = utils.get_roc_pr(*confusions_sum)[-2]
            cnt += 1
            print("[Cnt:%d] val_f1:%f" % (cnt, f1))

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
                    self.logger.will_write("[Train] epoch:%d loss:%f" % (epoch, recon_loss))

            if val_loader is not None:
                self.valid(epoch, val_loader)
            else:
                self.save(epoch)
        print("End Train\n")

    def _test_foward(self, input_, target_):
        input_ = input_.to(self.torch_device)
        output_ = self.G(input_)
        output_ = self.sigmoid(output_).type(torch.FloatTensor).numpy()
        target_ = target_.type(torch.FloatTensor).numpy()
        input_ = input_.type(torch.FloatTensor).numpy()
        return input_, output_, target_

    def valid(self, epoch, val_loader):
        self.G.eval()
        with torch.no_grad():
            # (tn, fp, fn, tp)
            confusions_sum = [0, 0, 0, 0]
            # F1(per dataset) , JSS, Dice(per image)
            dice, jss, cnt = 0, 0, 1

            for i, (input_, target_, _) in enumerate(val_loader):
                _, output_, target_ = self._test_foward(input_, target_)

                target_np = utils.slice_threshold(target_, 0.5)
                output_np = utils.slice_threshold(output_, 0.5)
                for b in range(target_.shape[0]):
                    target_f, output_f = target_np[b].flatten(), output_np[b].flatten()

                    # element wise sum
                    confusions = confusion_matrix(target_f, output_f).ravel()
                    confusions_sum += confusions
                    score = utils.get_roc_pr(*confusions)
                    dice += score[-2]
                    jss += score[-1]
                    cnt += 1

            tn, fp, fn, tp = confusions_sum
            precision, recall = tp / (tp + fp), tp / (fp + fn) 
            f05 = (5 * precision * recall) / (precision + (4 * recall))
            f2 = (5 * precision * recall) / ((4 * precision) + recall)
            jss /= cnt
            dice /= cnt

            metric = f05 + jss + dice
            if metric > self.best_metric:
                self.best_metric = metric
                self.save(epoch)

            self.logger.write("[Val] epoch:%d f2:%f jss:%f dice:%f" % (epoch, f2, jss, dice))

    def get_best_th(self, loader):
        y_true = np.array([])
        y_pred = np.array([])
        for i, (input_, target_, _) in enumerate(loader):
            input_, output_, target_ = self._test_foward(input_, target_)
            target_np = utils.slice_threshold(target_, 0.5)

            y_true = np.concatenate([y_true, target_np.flatten()], axis=0)
            y_pred = np.concatenate([y_pred, output_.flatten()],   axis=0)

        pr_values = np.array(precision_recall_curve(y_true, y_pred))

        # TODO : F0.5 Score
        f_best, th_best = -1, 0
        for precision, recall, threshold in zip(*pr_values):
            f05 = (5 * precision * recall) / (precision + (4 * recall))
            if f05 > f_best:
                f_best = f05
                th_best = threshold
        return f_best, th_best

    def test(self, test_loader, val_loader):
        print("\nStart Test")
        self.load()
        self.G.eval()
        with torch.no_grad():
            y_true = np.array([])
            y_pred = np.array([])

            confusions, cnt = [0, 0, 0, 0], 0
            f1_sum = 0
            for i, (input_, target_, f_name) in enumerate(test_loader):
                input_, output_, target_ = self._test_foward(input_, target_)

                target_np = utils.slice_threshold(target_, 0.5)
                output_np = utils.slice_threshold(output_, 0.5)

                y_true = np.concatenate([y_true, target_np.flatten()], axis=0)
                y_pred = np.concatenate([y_pred, output_.flatten()],   axis=0)

                for batch_idx in range(0, input_.shape[0]):
                    target_b = target_np[batch_idx, 0, :, :]
                    output_b = output_np[batch_idx, 0, :, :]
                    target_f, output_f = target_b.flatten(), output_b.flatten()

                    save_path = "%s/fold%s/%s" % (self.save_path, self.fold, f_name[batch_idx][:-4])
                    input_norm = input_[batch_idx, 0, :, :]
                    input_norm = (input_norm - input_norm.min()) / (input_norm.max() - input_norm.min())
                    utils.image_save(save_path, input_norm, target_b, output_b)

                    confusion = confusion_matrix(target_f, output_f).ravel()
                    confusions += confusion
                    scores = utils.get_roc_pr(*confusion)
                    self.logger.will_write("[Save] fname:%s sen:%f spec:%f prec:%f rec:%f f1:%f jss:%f" % (f_name[batch_idx][:-4], *scores))

                    f1_sum += scores[-2] # image per f1
                    cnt += 1

            roc_values = np.array(roc_curve(y_true, y_pred))
            pr_values = np.array(precision_recall_curve(y_true, y_pred))

            np.save("%s/fold%s/test_roc_values.npy" % (self.save_path, self.fold), roc_values)
            np.save("%s/fold%s/test_pr_values.npy" % (self.save_path, self.fold),  pr_values)

            tn, fp, fn, tp = confusions
            self.logger.will_write("[Save] tn:%d fp:%d fn:%d tp:%d"%(tn, fp, fn, tp))
            precision, recall = tp / (tp + fp), tp / (fp + fn) 
            f05 = (5 * precision * recall) / (precision + (4 * recall))
            f2 = (5 * precision * recall) / ((4 * precision) + recall)
            scores = utils.get_roc_pr(*confusions)
        self.logger.write("Best Threshold:%f sen:%f spec:%f prec:%f rec:%f f1:%f jss:%f dice:%f f05:%f f2:%f" % (0.5, *scores, f1_sum / float(cnt), f05, f2))
        print("End Test\n")
