import os
import json
import torch


class TrainingState:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.current_epoch = 1
        self.total_iters = 0
        self.best_ssim = 0.0
        self.best_psnr = 0.0
        self.best_lpips = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.load_state()

    def load_state(self):
        """Load training state from file if it exists"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.current_epoch = state.get('current_epoch', 1)
                self.total_iters = state.get('total_iters', 0)
                self.best_ssim = state.get('best_ssim', 0.0)
                self.best_psnr = state.get('best_psnr', 0.0)
                self.best_lpips = state.get('best_lpips', 0.0)
                self.best_epoch = state.get('best_epoch', 0)
                self.patience_counter = state.get('patience_counter', 0)
                print(
                    f'Loaded training state: epoch {self.current_epoch}, total iterations {self.total_iters}, best SSIM {self.best_ssim:.4f}, best PSNR {self.best_psnr:.4f}, best LPIPS {self.best_lpips:.4f} at epoch {self.best_epoch}')

    def save_state(self):
        """Save current training state to file"""
        state = {
            'current_epoch': self.current_epoch,
            'total_iters': self.total_iters,
            'best_ssim': float(self.best_ssim),
            'best_psnr': float(self.best_psnr),
            'best_lpips': float(self.best_lpips),
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def update_metrics(self, val_metrics, delta=0.0001, metric='ssim'):
        """Update metrics and return whether to continue training"""
        current_ssim = float(val_metrics['val_ssim'])
        current_psnr = float(val_metrics['val_psnr'])
        current_lpips = float(val_metrics['val_lpips'])
        current_metric = 0.0
        best_metric = 0.0
        
        if metric == 'ssim':
            print (f'Current SSIM: {current_ssim:.4f}, Best SSIM: {self.best_ssim:.4f}')
            current_metric = current_ssim
            best_metric = self.best_ssim
        elif metric == 'psnr':
            print (f'Current PSNR: {current_psnr:.4f}, Best PSNR: {self.best_psnr:.4f}')
            current_metric = current_psnr
            best_metric = self.best_psnr
        elif metric == 'lpips':
            print (f'Current LPIPS: {current_lpips:.4f}, Best LPIPS: {self.best_lpips:.4f}')
            current_metric = current_lpips
            best_metric = self.best_lpips

        if current_metric > best_metric + delta:
            self.best_ssim = current_ssim
            self.best_psnr = current_psnr
            self.best_lpips = current_lpips
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
            self.save_state()
            return True
        else:
            self.patience_counter += 1
            self.save_state()
            return False

    def increment_epoch(self):
        """Increment epoch counter and save state"""
        self.current_epoch += 1
        self.save_state()
