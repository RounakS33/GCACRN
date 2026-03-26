import torch
import itertools
from .base_model import BaseModel
from . import networks
from . import vgg
import numpy as np


class GCACRNModel(BaseModel, torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(
            no_dropout=False)
        parser.add_argument('--blurKernel', type=int, default=5,
                            help='maximum R for gaussian kernel')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.loss_names = ['idt_T', 'SSIM_T', 'mix_T', 'res', 'MP',
                           'G', 'T', 'idt_R', 'SSIM_R', 'mix_R', 'R', 'D_syn']

        if self.isTrain:
            self.visual_names = ['fake_Ts', 'fake_Rs']
        else:
            self.visual_names = ['fake_Ts', 'real_T', 'real_I', 'fake_Rs']

        if self.isTrain:
            self.model_names = ['G_T', 'G_R', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G_T', 'G_R']

        self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)

        # Define generator of synthesis net
        self.netG_T = networks.define_G(opt.input_nc * 3, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_R = networks.define_G(opt.input_nc * 3, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.criterionVgg = networks.VGGLoss(
            self.device, vgg=self.vgg, normalize=True)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netG_T.parameters(), self.netG_R.parameters()),
                                                 lr=opt.lr_G, betas=(opt.beta1, 0.999), weight_decay=0.001)
            self.optimizer_D = torch.optim.AdamW(itertools.chain(self.netD.parameters()),
                                                 lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=0.01)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.criterionIdt = torch.nn.SmoothL1Loss()
        self.criterionIdt2 = torch.nn.MSELoss()
        self.criterionSSIM = networks.SSIMLoss(data_range=1.0).to(self.device)
        # for synthetic images
        self.k_sz = np.linspace(0.8, self.opt.blurKernel, 80)

        self.t_h = None
        self.t_c = None
        self.r_h = None
        self.r_c = None

        self.fake_T = torch.zeros(
            self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Ts = [self.fake_T]

        # Pass invalid data
        self.trainFlag = True

        ''' We use both real-world data and synthetic data. If 'self.isNatural' is True, the data loaded is real-world
        image paris. Otherwise, we use 'self.syn' to synthesize data.'''
        self.isNatural = False
        self.syn = networks.SynData(self.device)
        self.real_I = None
        self.real_T = None
        self.real_T2 = None
        self.real_T4 = None
        self.alpha = None

    def set_input(self, input):
        """Unpack input data from the dataloader, perform necessary pre-processing steps and synthesize data.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        with torch.no_grad():
            if self.isTrain:
                if input['isNatural'][0] == 1:
                    self.isNatural = True
                else:
                    self.isNatural = False
                self.real_T2 = input['T2'].to(self.device)
                self.real_T4 = input['T4'].to(self.device)
                # Skip these procedures, if the data is from real-world.
                if not self.isNatural:
                    T = input['T'].to(self.device)
                    R = input['R'].to(self.device)
                    if torch.mean(T) * 1 / 2 > torch.mean(R):
                        self.trainFlag = False
                        return
                    _, R, I, alpha = self.syn(
                        T, R, self.k_sz)  # Synthesize data
                    self.alpha = round(alpha, 1)
                    if T.max() < 0.15 or R.max() < 0.15 or I.max() < 0.1:
                        self.trainFlag = False
                        return
                else:
                    I = input['I']
                    T = input['T']
            else:  # Test
                self.image_paths = input['B_paths']
                I = input['I']
                T = input['T']
                self.real_T2 = input['T2'].to(self.device)
                self.real_T4 = input['T4'].to(self.device)

        self.real_T = T.to(self.device)
        self.real_I = I.to(self.device)

    def init(self):
        self.t_h = None
        self.t_c = None
        self.r_h = None
        self.r_c = None
        self.fake_T = self.real_I.clone()
        self.fake_Ts = [self.fake_T]
        self.fake_R = torch.ones_like(self.real_I) * 0.1
        self.fake_Rs = [self.fake_R]

    def forward(self):
        self.init()
        i = 0
        while i <= 2:
            self.fake_T, self.t_h, self.t_c, self.fake_T2, self.fake_T4 = self.netG_T(
                torch.cat((self.real_I, self.fake_Ts[-1], self.fake_Rs[-1]), 1), self.t_h, self.t_c)
            self.fake_Ts.append(self.fake_T)
            self.fake_R, self.r_h, self.r_c, self.fake_R2, self.fake_R4 = self.netG_R(
                torch.cat((self.real_I, self.fake_Ts[-1], self.fake_Rs[-1]), 1), self.r_h, self.r_c)
            self.fake_Rs.append(self.fake_R)
            i += 1

        # clip operation in test and val
        if not self.isTrain:
            for i in range(len(self.fake_Ts)):
                self.fake_Ts[i] = torch.clamp(self.fake_Ts[i], min=0, max=1)
            for i in range(len(self.fake_Rs)):
                self.fake_Rs[i] = torch.clamp(self.fake_Rs[i], min=0, max=1)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D(self):
        self.loss_D_syn = self.backward_D_basic(
            self.netD, self.real_T, self.fake_T)
        self.loss_D_syn.backward()

    def compute_losses(self):
        """Calculates all losses and stores them as class attributes."""
        self.loss_mix_T = 0.0
        self.loss_mix_R = 0.0
        self.loss_idt_T = 0.0
        self.loss_idt_R = 0.0
        self.loss_SSIM_T = 0.0
        self.loss_SSIM_R = 0.0
        self.loss_res = 0.0
        self.loss_MP = 0.0
        iter_num = len(self.fake_Ts)

        sigma = 0.84
        real_I_r = torch.pow(self.real_I, 2.2)
        real_T_r = torch.pow(self.real_T, 2.2)

        for i in range(iter_num):
            if i > 0:
                T_r = torch.pow(self.fake_Ts[i], 2.2)
                R_r = torch.pow(self.fake_Rs[i], 2.2)
                self.loss_idt_T += self.criterionIdt(
                    self.fake_Ts[i], self.real_T) * np.power(sigma, iter_num - i)
                self.loss_SSIM_T += self.criterionSSIM(
                    self.fake_Ts[i], self.real_T) * np.power(sigma, iter_num - i)
                if not self.isNatural and self.isTrain:
                    self.loss_res += self.criterionIdt2(
                        real_I_r, (self.alpha * T_r + R_r)) * np.power(sigma, iter_num - i) * 5
                    self.loss_idt_R += self.criterionIdt(
                        R_r + real_T_r * self.alpha, real_I_r) * np.power(sigma, iter_num - i) * 2
                    self.loss_SSIM_R += self.criterionSSIM(
                        R_r + real_T_r * self.alpha, real_I_r) * np.power(sigma, iter_num - i) * 2

        self.loss_MP = 0.5 * (self.criterionVgg(self.fake_T, self.real_T) + 0.8 * self.criterionVgg(
            self.fake_T2, self.real_T2) + 0.6 * self.criterionVgg(self.fake_T4, self.real_T4))
        # self.loss_mix_T = 0.5 * self.loss_idt_T + 1.6 * self.loss_SSIM_T
        # self.loss_mix_R = 0.5 * self.loss_idt_R + 1.6 * self.loss_SSIM_R
        self.loss_mix_T = self.loss_idt_T
        self.loss_mix_R = self.loss_idt_R

        if self.isTrain:
            self.loss_G = self.criterionGAN(
                self.netD(self.fake_T), True) * 0.02
        else:
            self.loss_G = 0.0
            self.loss_D_syn = 0.0

        self.loss_T = self.loss_mix_T + self.loss_res + \
            self.loss_MP + self.loss_G

        self.loss_R = self.loss_mix_R

        self.loss = self.loss_R + self.loss_T

    def backward_G(self):
        """Calculates and backpropagates generator losses."""
        self.compute_losses()
        self.loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Pass invalid data
        if not self.trainFlag:
            self.trainFlag = True
            return False

        # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.netD], False)
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        torch.nn.utils.clip_grad_norm_(self.netG_T.parameters(), max_norm=0.25)
        torch.nn.utils.clip_grad_norm_(self.netG_R.parameters(), max_norm=0.25)
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()

        return True

    def print_parameter_status(self):
        """Print the number of trainable and frozen parameters for each network"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                trainable_params = sum(p.numel()
                                       for p in net.parameters() if p.requires_grad)
                frozen_params = sum(p.numel()
                                    for p in net.parameters() if not p.requires_grad)
                total_params = trainable_params + frozen_params
                print(f"\n{name} summary:")
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"Frozen parameters: {frozen_params:,}")
                print(f"Total parameters: {total_params:,}")
