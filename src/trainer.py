import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from model.networks import Generator, PatchD

from MMD import MMDLoss


class ImGANTrainer(nn.Module):
    def __init__(self, config):
        super(ImGANTrainer, self).__init__()

        self.netG_ab = Generator(config)
        self.netG_ba = Generator(config)
        self.netD_ab = PatchD(config['input_nc'], config['ndf'])
        self.netD_ba = PatchD(config['input_nc'], config['ndf'])

        self.optimizer_g = torch.optim.Adam(itertools.chain(self.netG_ab.parameters(), self.netG_ba.parameters()),
                                            lr=config['lr'], betas=(config['beta1'], 0.999))
        self.optimizer_d = torch.optim.Adam(itertools.chain(self.netD_ab.parameters(), self.netD_ba.parameters()),
                                            lr=config['lr'], betas=(config['beta1'], 0.999))
        # criterion
        self.criteritionGAN = nn.BCELoss()
        self.criteritioL1 = nn.L1Loss()
        self.criteritiommd = MMDLoss()
        # labels
        self.real_label = 1.
        self.fake_label = 0.

        # losses
        self.loss_names = ['loss_D', 'loss_G', 'loss_cycle_aba', 'loss_cycle_bab', 'loss_mmd']

    def dis_basic(self, netD, real, fake):
        # real
        output_real = netD(real)
        real_label = Variable(torch.FloatTensor(output_real.size()).fill_(self.real_label).cuda())
        loss_D_real = self.criteritionGAN(output_real, real_label)
        # fake
        output_fake = netD(fake)
        fake_label = Variable(torch.FloatTensor(output_fake.size()).fill_(self.fake_label).cuda())
        loss_D_fake = self.criteritionGAN(output_fake, fake_label)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        return loss_D

    def dis_upodate(self, input_a, input_b):
        self.optimizer_d.zero_grad()
        latent_a, common_a = self.netG_ab.encoder(input_a)
        latent_b, common_b = self.netG_ba.encoder(input_b)
        # decode
        input_ab = latent_a - common_a + common_b
        fake_ab = self.netG_ab.decoder(input_ab)
        input_ba= latent_b - common_b + common_a
        fake_ba = self.netG_ba.decoder(input_ba)
        # compute the loss
        self.loss_D = self.dis_basic(self.netD_ab, input_b, fake_ab.detach()) + \
                      self.dis_basic(self.netD_ba, input_a, fake_ba.detach())
        self.optimizer_d.step()

    def gen_update(self, input_a, input_b):
        self.optimizer_g.zero_grad()
        # encode
        latent_a, common_a = self.netG_ab.encoder(input_a)
        latent_b, common_b = self.netG_ba.encoder(input_b)
        # decode
        input_ab = latent_a - common_a + common_b
        fake_ab = self.netG_ab.decoder(input_ab)
        input_ba = latent_b - common_b + common_a
        fake_ba = self.netG_ba.decoder(input_ba)
        # cycle
        latent_a_rec, common_a_rec = self.netG_ab.encoder(fake_ba)
        latent_b_rec, common_b_rec = self.netG_ba.encoder(fake_ab)
        input_ab_rec = latent_a_rec - common_a_rec + common_b_rec
        fake_bab = self.netG_ab.decoder(input_ab_rec)
        input_ba_rec = latent_b_rec - common_b_rec + common_a_rec
        fake_aba = self.netG_ba.decoder(input_ba_rec)

        # compute the loss
        output_ab = self.netD_ab(fake_ab)
        label_ab = Variable(torch.FloatTensor(output_ab.size()).fill_(self.real_label).cuda())
        self.loss_g_ab = self.criteritionGAN(output_ab, label_ab)
        output_ba = self.netD_ba(fake_ba)
        label_ba = Variable(torch.FloatTensor(output_ba.size()).fill_(self.real_label).cuda())
        self.loss_g_ba = self.criteritionGAN(output_ba, label_ba)
        # cycle loss
        self.loss_cycle_aba = self.criteritioL1(fake_aba, input_a)
        self.loss_cycle_bab = self.criteritioL1(fake_bab, input_b)
        # mmd loss for  latena and latenb
        self.loss_mmd = self.criteritiommd(latent_a - common_a, latent_b - common_b)
        self.loss_G = self.loss_g_ab + self.loss_g_ba + self.loss_cycle_aba + self.loss_cycle_bab + self.loss_mmd
        # self.loss_G = self.loss_g_ab
        self.loss_G.backward()

        self.optimizer_g.step()

    def test(self, input_a, input_b):
        self.eval()
        # encode
        latent_a, common_a = self.netG_ab.encoder(input_a)
        latent_b, common_b = self.netG_ba.encoder(input_b)
        # decode
        input_ab = latent_a - common_a + common_b
        fake_ab = self.netG_ab.decoder(input_ab)
        input_ba = latent_b - common_b + common_a
        fake_ba = self.netG_ba.decoder(input_ba)
        latent_a_rec, common_a_rec = self.netG_ab.encoder(fake_ba)
        latent_b_rec, common_b_rec = self.netG_ba.encoder(fake_ab)
        input_ab_rec = latent_a_rec - common_a_rec + common_b_rec
        fake_bab = self.netG_ab.decoder(input_ab_rec)
        input_ba_rec = latent_b_rec - common_b_rec + common_a_rec
        fake_aba = self.netG_ba.decoder(input_ba_rec)
        self.train()
        return input_a, input_b, fake_ab, fake_ba, fake_bab, fake_aba

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, name)
        return errors_ret

    def save_model(self, outf, iters):
        pass

    def resume(self):
        pass

