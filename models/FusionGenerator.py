from model.Basic_blocks import *


class Conv_residual_conv(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(Conv_residual_conv, self).__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim

        self.conv_1 = conv_block(self.in_dim, self.out_dim, act_fn)
        self.conv_2 = conv_block_3(self.out_dim, self.out_dim, act_fn)
        self.conv_3 = conv_block(self.out_dim, self.out_dim, act_fn)

    def forward(self, input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res    = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, output, out_clamp=None):
        super(FusionGenerator, self).__init__()

        self.output = output
        self.out_clamp = out_clamp
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ELU(inplace=True)

        print("\n------Initiating FusionNet------\n")

        # encoder
        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = conv_block(self.out_dim, self.out_dim, act_fn, 2)
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = conv_block(self.out_dim * 2, self.out_dim * 2, act_fn, 2)
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = conv_block(self.out_dim * 4, self.out_dim * 4, act_fn, 2)
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = conv_block(self.out_dim * 8, self.out_dim * 8, act_fn, 2)

        # bridge
        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder
        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)

        if output == "sin2res":
            self.out_2 = nn.Linear()  # Sinogram -> Residual : Linear
        elif output == "img2res":
            self.out_2 = nn.Tanh()    # Image -> Residual : Tanh
        elif output == "img2img":
            self.out_2 = nn.Sigmoid() # Image -> Image : Sigmoid
        else:
            raise NotImplementedError() # TODO :

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1  = self.deconv_1(bridge)
        skip_1    = (deconv_1 + down_4) / 2
        up_1      = self.up_1(skip_1)

        deconv_2  = self.deconv_2(up_1)
        skip_2    = (deconv_2 + down_3) / 2
        up_2      = self.up_2(skip_2)

        deconv_3  = self.deconv_3(up_2)
        skip_3    = (deconv_3 + down_2) / 2
        up_3      = self.up_3(skip_3)

        deconv_4  = self.deconv_4(up_3)
        skip_4    = (deconv_4 + down_1) / 2
        up_4      = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_2(out)
        if self.out_clamp is not None:
            out = torch.clamp(out, min=self.out_clamp[0], max=self.out_clamp[1])

        return out


class FusionGenerator:

    def __init__(self, arg, recon_loss, mse_loss=None):
        self.output = arg.output
        self.model_type = arg.model
        self.epoch = epoch
        self.batch_size = arg.batch_size
        self.recon_loss = recon_loss
        self.mse_loss = mse_loss

        self.start_epoch = 0
        self.best_mse = 1

        self.G = FusionNet(1, 1, arg.ngf, arg.output, arg.clamp).cuda()
        if len(arg.gpus) > 1:
            self.G = nn.DataParallel(self.G).cuda()
            
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=(arg.beta1, arg.beta2))

        self.save_path = util.get_save_dir(arg)
        if len(arg.load) != 0:
            self._load()


    def save(self):
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        torch.save({"model_type" : self.model_name,
                    "start_epoch" : self.start_epoch,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.opim.state_dict()
                    }, os.path.join(self.save_path) + "models.pth.tar")

    def _load(load_path):
        checkpoint = torch.load(load_path)
        if checkpoint['model_type'] != "fusion":
            raise Exception("model_type is %s"%(checkpoint["model_type"]))
        
        self.G.load_state_dict(checkpoint['network'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['start_epoch']


    def train(self, train_loader, val_loader=None):
        print("\nStart Train")
        best_loss = None
        # TODO : save ??
        for epoch in range(self.start_epoch, self.epoch):
            for i, (input_, target_, _) in enumerate(train_loader):
                self.G.train()

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
            
                if i % 100 === 0:
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

            input_  =  input_.cpu.data[0].numpy()
            target_ = target_.cpu.data[0].numpy()
            output_ = output_.cpu.data[0].numpy()

            mse_loss += (utils.mse(output_, target_))
            
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

