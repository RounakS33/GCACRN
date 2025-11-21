import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import piq


def test(model, opt, test_dataloader, webpage):
    """Run testing with proper TorchMetrics usage"""
    if test_dataloader is None:
        return {'test_ssim': 0.0, 'test_psnr': 0.0, 'test_lpips': 0.0}

    model.eval()

    # Initialize metrics
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0).to(model.device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(model.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg', normalize=True).to(model.device)

    losses = {k: 0.0 for k in model.loss_names}

    i = 0

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc='Testing')
        for data in pbar:
            i += 1
            model.set_input(data)
            model.forward()
            model.compute_losses()
            losses_dict = model.get_current_losses()
            for k, v in losses_dict.items():
                losses[k] += v
            model.compute_visuals()
            # visuals = model.get_current_visuals()
            # image_paths = model.get_image_paths()
            fake = model.fake_T.clamp(0, 1)
            real = model.real_T.clamp(0, 1)
            # Update metrics
            ssim_metric.update(fake, real)
            psnr_metric.update(fake, real)
            lpips_metric.update(fake, real)

            # if i % opt.display_freq == 0:
            #     save_images(webpage, visuals, image_paths,
            #                 aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # Compute averages
    avg_ssim = ssim_metric.compute()
    avg_psnr = psnr_metric.compute()
    avg_lpips = lpips_metric.compute()

    for k, v in losses.items():
        losses[k] = v / len(test_dataloader)

    # Reset states
    ssim_metric.reset()
    psnr_metric.reset()
    lpips_metric.reset()

    return {
        'test_ssim': avg_ssim,
        'test_psnr': avg_psnr,
        'test_lpips': avg_lpips,
        'losses': losses
    }


def main():
    opt = TestOptions().parse()   # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1  # batch size of 1 for testing
    opt.phase = 'test'  # set phase to test
    opt.isTrain = False
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    opt.display_freq = 1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (
        opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
        opt.name, opt.phase, opt.epoch))

    # Run testing and get SSIM scores
    test_metrics = test(model, opt, dataset.test_dataloader, webpage)
    print(f'\nTest SSIM: {test_metrics["test_ssim"]:.4f}')
    print(f'Test PSNR: {test_metrics["test_psnr"]:.4f}')
    print(f'Test LPIPS: {test_metrics["test_lpips"]:.4f}')

    # create or apend to log file the losses and metrics
    test_loss_log_file = os.path.join(
        opt.results_dir, opt.name, 'test_loss_log.txt')
    test_metric_log_file = os.path.join(
        opt.results_dir, opt.name, 'test_metric_log.txt')
    # format: epoch,loss_idt_T,loss_SSIM_T,loss_mix_T,loss_res,loss_MP,loss_G,loss_T,loss_idt_R,loss_SSIM_R,loss_mix_R,loss_R,loss_D_syn
    with open(test_loss_log_file, 'a') as f:
        loss_items = test_metrics['losses'].items()
        loss_str = ','.join([f'{v:.4f}' for k, v in loss_items])
        f.write(f'{opt.epoch}, {loss_str}\n')
    # format: epoch,test_ssim,test_psnr,test_lpips
    with open(test_metric_log_file, 'a') as f:
        f.write(
            f'{opt.epoch},{test_metrics["test_ssim"]:.4f},{test_metrics["test_psnr"]:.4f},{test_metrics["test_lpips"]:.4f}\n')

    webpage.save()  # save the HTML


if __name__ == '__main__':
    main()
