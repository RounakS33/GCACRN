import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from training_state import TrainingState


def validate(model, val_dataloader):
    """Run validation using TorchMetrics"""
    if val_dataloader is None:
        return {'val_ssim': 0.0, 'val_psnr': 0.0, 'val_lpips': 0.0, 'losses': {}}

    model.eval()
    model.opt.phase = 'val'  # Set phase to validation
    model.isTrain = False  # Ensure model is in evaluation mode

    # Initialize metrics (accumulate automatically)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0).to(model.device)
    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0).to(model.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg', normalize=True).to(model.device)

    losses = {k: 0.0 for k in model.loss_names}

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc='Validation')
        for data in pbar:
            model.set_input(data)
            model.forward()
            model.compute_losses()
            losses_dict = model.get_current_losses()
            for k, v in losses_dict.items():
                losses[k] += v
            fake = model.fake_T.clamp(0, 1)
            real = model.real_T.clamp(0, 1)
            ssim_metric.update(fake, real)
            psnr_metric.update(fake, real)
            lpips_metric.update(fake, real)

    # Compute averages
    avg_ssim = ssim_metric.compute()
    avg_psnr = psnr_metric.compute()
    avg_lpips = lpips_metric.compute()

    for k, v in losses.items():
        losses[k] = v / len(val_dataloader)

    # Reset metrics for safety
    ssim_metric.reset()
    psnr_metric.reset()
    lpips_metric.reset()

    model.opt.phase = 'train'  # Reset phase back to training
    model.isTrain = True  # Reset model to training mode

    return {
        'val_ssim': avg_ssim,
        'val_psnr': avg_psnr,
        'val_lpips': avg_lpips,
        'losses': losses
    }


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    train_size = len(dataset.train_dataset)
    val_size = len(dataset.val_dataset)
    print('The number of training images = %d' % train_size)
    print('The number of validation images = %d' % val_size)

    model = create_model(opt)
    model.setup(opt)

    # Initialize training state
    training_state = TrainingState(os.path.join(opt.checkpoints_dir, opt.name))
    total_iters = training_state.total_iters  # Load total iterations from state

    # Create or append to log file
    train_loss_log_file = os.path.join(
        opt.checkpoints_dir, opt.name, 'train_loss_log.txt')
    train_metric_log_file = os.path.join(
        opt.checkpoints_dir, opt.name, 'train_metric_log.txt')
    val_loss_log_file = os.path.join(
        opt.checkpoints_dir, opt.name, 'val_loss_log.txt')
    val_metric_log_file = os.path.join(
        opt.checkpoints_dir, opt.name, 'val_metric_log.txt')
    if not opt.continue_train and training_state.current_epoch == 1:
        with open(train_loss_log_file, "w") as ft:
            loss_columns = ','.join(['loss_' + k for k in model.loss_names])
            ft.write('epoch,' + loss_columns + '\n')
        with open(val_loss_log_file, "w") as fv:
            loss_columns = ','.join(['loss_' + k for k in model.loss_names])
            fv.write('epoch,' + loss_columns + '\n')
        with open(train_metric_log_file, "w") as fm:
            fm.write('epoch,total_iter,ssim,psnr,lpips\n')
        with open(val_metric_log_file, "w") as fvm:
            fvm.write('epoch,ssim,psnr,lpips\n')

    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0).to(model.device)
    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0).to(model.device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg', normalize=True).to(model.device)

    model.print_parameter_status()

    visualizer = Visualizer(opt)

    while training_state.current_epoch <= opt.niter + opt.niter_decay:
        model.isTrain = True
        model.train()
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        ssim_metric.reset()
        psnr_metric.reset()
        lpips_metric.reset()
        losses = {k: 0.0 for k in model.loss_names}
        invalid = 0

        # Create a single progress bar for training
        pbar = tqdm(dataset.train_dataloader,
                    desc=f'Epoch {training_state.current_epoch}/{opt.niter + opt.niter_decay}')
        for i, data in enumerate(pbar):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            visualizer.reset()
            total_iters += 1
            epoch_iter += 1

            model.set_input(data)
            is_optimized = model.optimize_parameters()

            if not is_optimized:
                invalid += 1
                continue

            losses_dict = model.get_current_losses()
            for k, v in losses_dict.items():
                losses[k] += v

            fake = model.fake_T.clamp(0, 1)
            real = model.real_T.clamp(0, 1)
            ssim_metric.update(fake, real)
            psnr_metric.update(fake, real)
            lpips_metric.update(fake, real)

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), total_iters, save_result)

        avg_ssim = ssim_metric.compute()
        avg_psnr = psnr_metric.compute()
        avg_lpips = lpips_metric.compute()

        for k, v in losses.items():
            losses[k] = v / (len(dataset.train_dataloader) - invalid)

        with open(train_loss_log_file, "a") as f:
            f.write(
                f'{training_state.current_epoch},{",".join([f"{losses[k]:.4f}" for k in model.loss_names])}\n')

        with open(train_metric_log_file, "a") as fm:
            fm.write(
                f'{training_state.current_epoch},{total_iters},{avg_ssim:.4f},{avg_psnr:.4f},{avg_lpips:.4f}\n')

        # Run validation after each epoch
        val_metrics = validate(model, dataset.val_dataloader)
        current_ssim, current_psnr, current_lpips = val_metrics[
            'val_ssim'], val_metrics['val_psnr'], val_metrics['val_lpips']
        print(
            f'\nValidation SSIM: {current_ssim:.4f}, PSNR: {current_psnr:.4f}, LPIPS: {current_lpips:.4f}')

        loss_arr = [
            f"{val_metrics['losses'][k]:.4f}" for k in model.loss_names]
        with open(val_loss_log_file, "a") as f:
            f.write(
                f'{training_state.current_epoch},{",".join(loss_arr)}\n')

        with open(val_metric_log_file, "a") as f:
            f.write(
                f"{training_state.current_epoch},{current_ssim:.4f},{current_psnr:.4f},{current_lpips:.4f}\n")

        if training_state.update_metrics(val_metrics, opt.delta):
            print(
                f'Saving best model with SSIM: {current_ssim:.4f}, PSNR: {current_psnr:.4f}, LPIPS: {current_lpips:.4f}')
            model.save_networks('best')
        else:
            if training_state.patience_counter >= opt.patience:
                print(
                    f'Early stopping triggered. Best SSIM: {training_state.best_ssim:.4f} at epoch {training_state.best_epoch}.')
                break

        # Save latest model and update state
        print('saving the latest model at the end of epoch %d, iters %d' %
              (training_state.current_epoch, total_iters))
        model.save_networks('latest')
        training_state.total_iters = total_iters  # Update total iterations in state
        training_state.save_state()

        # Clear GPU memory after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (training_state.current_epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        # Increment epoch counter
        training_state.increment_epoch()
