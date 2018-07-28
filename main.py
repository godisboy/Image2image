import random
import sys
import os
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

from src.trainer import ImGANTrainer
from data.dataset import DATASET
from utils.tools import get_config

parser = OptionParser()
parser.add_option('--config', type=str, help="training configuration")
parser.add_option('--cuda', action='store_true', help='enables cuda')
parser.add_option('--gpu_ids', default=0, type=int, help='enables cuda')
parser.add_option('--manualSeed', type=int, help='manual seed')


def main(argv):
    (opt, args) = parser.parse_args(argv)
    print(opt)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print ("random seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.cuda.set_device(opt.gpu_ids)
    cudnn.benchmark = True
    config = get_config(opt.config)
    # loading data sets
    datasetA = DATASET(os.path.join(config['dataPath'], 'trainA'), config['loadSize'], config['fineSize'], config['flip'])
    datasetB = DATASET(os.path.join(config['dataPath'], 'trainB'), config['loadSize'], config['fineSize'], config['flip'])
    loader_A = torch.utils.data.DataLoader(dataset=datasetA,
                                           batch_size=config['batchSize'],
                                           shuffle=True,
                                           num_workers=config['num_workers'])
    loaderA = iter(loader_A)
    loader_B = torch.utils.data.DataLoader(dataset=datasetB,
                                           batch_size=config['batchSize'],
                                           shuffle=True,
                                           num_workers=config['num_workers'])
    loaderB = iter(loader_B)
    # define the trainer
    trainer = ImGANTrainer(config)
    print(trainer.netG_ab, trainer.netD_ab)
    if opt.cuda:
        trainer.cuda()
    for iteration in range(1, config['niter'] + 1):
        try:
            imgA = loaderA.next()
            imgB = loaderB.next()
        except StopIteration:
            loaderA, loaderB = iter(loader_A), iter(loader_B)
            imgA = loaderA.next()
            imgB = loaderB.next()
        image_A = Variable(imgA.cuda())
        image_B = Variable(imgB.cuda())

        trainer.dis_upodate(image_A, image_B)
        trainer.gen_update(image_A, image_B)

        if iteration % 100 == 0:
            trainer.get_current_losses()
            losses = trainer.get_current_losses()
            message = '([%d/%d][%d/%d]) ' % (
                iteration, config['niter'], len(loader_A), len(loader_B))
            for k, v in losses.items():
                message += '%s: %.6f ' % (k, v)
            print(message)

            input_a, input_b, fake_ab, fake_ba, fake_bab, fake_aba = trainer.test(image_A, image_B)
            vutils.save_image(image_A.data,
                              '%s/realA_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)
            vutils.save_image(image_B.data,
                              '%s/realB_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)
            vutils.save_image(fake_aba.data,
                              '%s/recA_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)
            vutils.save_image(fake_bab.data,
                              '%s/recB_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)
            vutils.save_image(fake_ab.data,
                              '%s/AB_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)
            vutils.save_image(fake_ba.data,
                              '%s/BA_niter_%03d_1.png' % (config['outf'], iteration),
                              normalize=True)


if __name__ == '__main__':
    main(sys.argv)

