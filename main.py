import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
from reconstru_visual import visua_and_save
from mask_obtain import mask_obtain
import deal_sst_util
from easydict import EasyDict
from anomaly_mask import compute_anomaly_mask
from utils import *
from diffusion import AsyncDiffusion
from model import Image64Net, ExponentialMovingAverage

class AsyncDSBRunner:
    def __init__(self, opt, log):
        self.opt = opt
        self.log = log
        self.ckpt_dir = getattr(opt, 'ckpt_dir', './checkpoints_async_sb')
        # Best metrics per epoch/batch
        self.mim_rmse = float('inf')
        self.mim_mse = float('inf')
        self.mim_mae = float('inf')
        self.mim_r2 = -float('inf')
        self.mim_rmse_epoch = 0
        self.mim_mse_epoch = 0
        self.mim_mae_epoch = 0
        self.mim_r2_epoch = 0
        self.mim_rmse_batch = 0
        self.mim_mse_batch = 0
        self.mim_mae_batch = 0
        self.mim_r2_batch = 0
        
        # Average Best Metric
        self.min_mean_rmse = float('inf')
        self.min_mean_mse = float('inf')
        self.min_mean_mae = float('inf')
        self.max_mean_r2 = -float('inf')
        self.min_mean_rmse_epoch = 0
        self.min_mean_mse_epoch = 0
        self.min_mean_mae_epoch = 0
        self.max_mean_r2_epoch = 0
        self.tau_min = opt.tau_min
        self.tau_max = opt.tau_max
        
        # Initialize diffusion model
        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        self.betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = AsyncDiffusion(betas, opt.device, tau_min=opt.tau_min, tau_max=opt.tau_max)
        log.info(f"[Diffusion] Built AsyDM-SST diffusion: steps={len(betas)}!")
        
        # Initialize the main network
        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image64Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.net.to(opt.device)
        self.ema.to(opt.device)
      
        
    
    def _model_state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    def _ema_state_dict_cpu(self):
        return {k: v.detach().cpu().clone() for k, v in self.ema.export_state_dict(self.net).items()}

    def save_checkpoint(self, tag, epoch, metrics=None):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        payload = {
            'tag': tag,
            'epoch': epoch,
            'bridge_type': 'async_sb',
            'model': self._model_state_dict_cpu(),
            'ema_model': self._ema_state_dict_cpu(),
            'metrics': metrics or {},
            'opt': dict(self.opt),
        }
        ckpt_path = os.path.join(self.ckpt_dir, f'{tag}.pth')
        torch.save(payload, ckpt_path)
        self.log.info(f"[Checkpoint] Saved {tag} checkpoint to {ckpt_path}")
        return ckpt_path

    def compute_label(self, step, x0, xt, anomaly_mask=None):
        if anomaly_mask is not None:
            std_fwd = self.diffusion.get_std_fwd_pixel(step, anomaly_mask, xdim=x0.shape[1:])
        else:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
    
    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False, anomaly_mask=None):
        if anomaly_mask is not None:
            std_fwd = self.diffusion.get_std_fwd_pixel(step, anomaly_mask, xdim=xt.shape[1:])
        else:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1., 1.)
        return pred_x0

    def train(self, opt, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        optimizer = optim.AdamW(self.net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
        mse = nn.MSELoss()
        
        # Training Metrics Record
        train_loss = {'reconstruct': [], 'anomaly_constraint': []}
        epoch_avg_loss = {'reconstruct': np.zeros(opt.epoches), 'anomaly_constraint': np.zeros(opt.epoches)}
        
        for epoch in range(opt.epoches):
            self.net.train()
            epoch_reconstruct = 0.0
            epoch_anomaly_constraint = 0.0
            
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                batch_x = batch_x.to(opt.device)
                batch_y = batch_y.to(opt.device)
                optimizer.zero_grad()

                # data
                x1_orig = batch_x[:, 6:7]  # corrupted image
                x0 = batch_y[:, 6:7]  # target
                x3 = batch_y[:, 7:8]  # weekly mean
                mask_orig = mask_obtain("mask", opt.mask_type, opt.corrup_rate, 
                                      mask_num=1, batch_size=batch_x.shape[0]).to(opt.device)
               # mask_orig = 0: missing       
                x1_com = (x1_orig * mask_orig) + (x3 * (1 - mask_orig))
                anomaly_mask = compute_anomaly_mask(x1_com, x3)
                anomaly_mask = 1. - anomaly_mask
                step = torch.randint(0, opt.interval, (x0.shape[0],), device=opt.device, dtype=torch.long)
                xt = self.diffusion.q_sample(step, x0, x1_com, anomaly_mask=anomaly_mask)
                label = self.compute_label(step, x0, xt, anomaly_mask).float()
                pred = self.net(xt, step, cond=x3).float()
                pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=anomaly_mask, clip_denoise=opt.clip_denoise)
                #Loss Term
                loss_true = mse(pred, label)
                true_anomaly = x0 - x3
                pred_anomaly = pred_x0 - x3
                loss_anomaly = mse(pred_anomaly, true_anomaly)

                transition_point = opt.epoches * 0.25
                w_true = 0.6 + 0.4 * min(1.0, epoch / transition_point)
                w_remain = max(0.0, 1.0 - w_true)
                w_anomaly = w_remain

                #  total loss
                total_loss = w_true * loss_true + w_anomaly * loss_anomaly
                total_loss.backward()
                optimizer.step()
                self.ema.update()

                # Record loss
                train_loss['reconstruct'].append(loss_true.item())
                train_loss['anomaly_constraint'].append(loss_anomaly.item())
                epoch_reconstruct += loss_true.item()
                epoch_anomaly_constraint += loss_anomaly.item()
                
                #  Visualization
                if epoch == opt.visual_epoch:
                    pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                    visua_and_save('Train', epoch, i, x1_orig, f'miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, x1_com, 'x1_com', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, x0, 'ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                    pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                    visua_and_save('Train', epoch, i, x1_com, 'x1_com', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
            
            # Calculate the average loss per epoch
            epoch_avg_loss['reconstruct'][epoch] = epoch_reconstruct / len(train_loader)
            print(f'Train Epoch: {epoch} | Reconstruct_batch: {train_loss["reconstruct"][-1]:.6f} | Reconstruct_ave: {epoch_avg_loss["reconstruct"][epoch]:.6f}')
            
            # Plot the loss curve
            # if epoch % opt.visual_epoch == 0:
            #     loss_curve = np.asarray(epoch_avg_loss['reconstruct'][:epoch + 1], dtype=float)
            #     epoch_idx = np.arange(1, len(loss_curve) + 1)
            #     warmup_epochs = min(5, len(loss_curve))
            #     alpha_fast = 0.65
            #     alpha_stable = 0.28
            #     smooth_curve = np.empty_like(loss_curve)
            #     smooth_curve[0] = loss_curve[0]
            #     for i in range(1, len(loss_curve)):
            #         alpha = alpha_fast if i < warmup_epochs else alpha_stable
            #         smooth_curve[i] = alpha * loss_curve[i] + (1 - alpha) * smooth_curve[i - 1]

            #     with plt.style.context('seaborn-v0_8-whitegrid'):
            #         fig, ax = plt.subplots(figsize=(10, 5.6), dpi=220)
            #         ax.plot(
            #             epoch_idx,
            #             loss_curve,
            #             color='#9AA5B1',
            #             linewidth=1.6,
            #             alpha=0.7,
            #             label='Epoch Avg Loss'
            #         )
            #         ax.plot(
            #             epoch_idx,
            #             smooth_curve,
            #             color='#1F6FEB',
            #             linewidth=2.4,
            #             label='EWMA-Smoothed Loss'
            #         )

            #         ax.set_title('Training Reconstruction Loss', fontsize=14, pad=10)
            #         ax.set_xlabel('Epoch', fontsize=12)
            #         ax.set_ylabel('Loss', fontsize=12)
            #         ax.tick_params(axis='both', labelsize=10)
            #         ax.margins(x=0.02)
            #         ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.45)
            #         ax.legend(loc='upper right', frameon=False, fontsize=10)
            #         fig.tight_layout()
            #         fig.savefig(f'epoch_loss_epoch{epoch}.png', bbox_inches='tight')
            #         plt.close(fig)

            
            # validate
            self.validate(epoch, val_loader, opt)

        final_metrics = {
            'rmse': self.min_mean_rmse,
            'mse': self.min_mean_mse,
            'mae': self.min_mean_mae,
            'r2': self.max_mean_r2,
        }
        self.save_checkpoint('last', opt.epoches - 1, metrics=final_metrics)

    def validate(self, epoch, val_loader, opt):
        self.net.eval()
        total_mse, total_rmse, total_mae, total_r2 = [], [], [], []
        test_loss = np.zeros(opt.epoches, dtype=float)
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(val_loader)):
                batch_x = batch_x.to(opt.device)
                batch_y = batch_y.to(opt.device)
                x1_orig = batch_x[:, 6:7].float()
                x0 = batch_y[:, 6:7].float()
                x3 = batch_y[:, 7:8].float()
                mask_orig = mask_obtain("mask", opt.mask_type, opt.corrup_rate, mask_num=1, batch_size=batch_x.shape[0]).to(opt.device)
                x1_com = (x1_orig * mask_orig) + (x3 * (1 - mask_orig))
                anomaly_mask = compute_anomaly_mask(x1_com, x3)
                anomaly_mask = 1. - anomaly_mask
                xs, pred_x0s = self.run_async_sampling(opt, x1_com, cond=x3, anomaly_mask=anomaly_mask, clip_denoise=opt.clip_denoise, nfe=30,log_count=30)
                reconstructed_image = pred_x0s[:,-1]
                # print("re",reconstructed_image.shape)
                 #  Visualization
                if epoch == opt.visual_epoch:
                    visua_and_save('Test', epoch, i, x1_orig, f'valid_miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, x3, f'week', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, x1_com, f'x1_com', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, x0, 'valid_ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                    visua_and_save('Test', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                
                # evaluation
                difference = reconstructed_image - x0
                mask = (mask_orig == 0).float()
                mse_score = torch.sum((difference * mask) ** 2) / torch.sum(mask)
                total_mse.append(mse_score.item())

                rmse_score = torch.sqrt(mse_score)
                total_rmse.append(rmse_score.item())

                mae_score = torch.sum(torch.abs(difference * mask)) / torch.sum(mask)
                total_mae.append(mae_score.item())

                SSE = torch.sum((difference ** 2) * mask)
                SST = torch.sum((x0 - (torch.sum(x0 * mask) / (torch.sum(mask))) * mask) ** 2)
                R2_score = 1 - SSE / SST
                total_r2.append(R2_score.item())
            
                if epoch >= opt.visual_epoch:
                    if self.mim_rmse >= rmse_score.item():
                        self.mim_rmse = rmse_score.item()
                        self.mim_rmse_epoch = epoch
                        self.mim_rmse_batch = i + 1
                        visua_and_save('Test', epoch, i, reconstructed_image, 'valid_min_rmse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mse >= mse_score.item():
                        self.mim_mse = mse_score.item()
                        self.mim_mse_epoch = epoch
                        self.mim_mse_batch = i + 1
                        visua_and_save('Test', epoch, i, reconstructed_image, 'valid_min_mse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mae >= mae_score.item():
                        self.mim_mae = mae_score.item()
                        self.mim_mae_epoch = epoch
                        self.mim_mae_batch = i + 1
                        visua_and_save('Test', epoch, i, reconstructed_image, 'valid_min_mae_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_r2 <= R2_score.item():
                        self.mim_r2 = R2_score.item()
                        self.mim_r2_epoch = epoch
                        self.mim_r2_batch = i + 1
                        visua_and_save('Test', epoch, i, reconstructed_image, 'valid_min_r2_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)

            print('Valid Epoch: {}\t RMSE Loss: {:.8f}\t MSE Loss: {:.8f} \t MAE Loss: {:.8f} \t R2 Loss: {:.8f}\t'
                  .format(epoch, np.mean(total_rmse), np.mean(total_mse), np.mean(total_mae), np.mean(total_r2)))

            # global min_mean_rmse, min_mean_mse, min_mean_mae, max_mean_r2
            # global min_mean_rmse_epoch, min_mean_mse_epoch, min_mean_mae_epoch, max_mean_r2_epoch
            current_metrics = {
                'rmse': float(np.mean(total_rmse)),
                'mse': float(np.mean(total_mse)),
                'mae': float(np.mean(total_mae)),
                'r2': float(np.mean(total_r2)),
            }
            if epoch >= opt.visual_epoch:
                best_rmse_updated = False
                if self.min_mean_rmse >= current_metrics['rmse']:
                    self.min_mean_rmse = current_metrics['rmse']
                    self.min_mean_rmse_epoch = epoch
                    best_rmse_updated = True
                if self.min_mean_mse >= current_metrics['mse']:
                    self.min_mean_mse = current_metrics['mse']
                    self.min_mean_mse_epoch = epoch
                if self.min_mean_mae >= current_metrics['mae']:
                    self.min_mean_mae = current_metrics['mae']
                    self.min_mean_mae_epoch = epoch
                if self.max_mean_r2 <= current_metrics['r2']:
                    self.max_mean_r2 = current_metrics['r2']
                    self.max_mean_r2_epoch = epoch
                if best_rmse_updated:
                    self.save_checkpoint('best_rmse', epoch, metrics=current_metrics)

            test_loss[epoch] = np.mean(total_mse)

        result = [self.min_mean_rmse, self.min_mean_rmse_epoch, self.min_mean_mse, self.min_mean_mse_epoch,
                 self.min_mean_mae, self.min_mean_mae_epoch, self.max_mean_r2, self.max_mean_r2_epoch]
        with open(opt.path, 'w') as file:
            file.write("best: Rmse:" + str(result[0]) + "[Epoch:" + str(result[1]) + "] "
                       + "mse:" + str(result[2]) + "[Epoch:" + str(result[3]) + "] "
                       + "mae:" + str(result[4]) + "[Epoch:" + str(result[5]) + "] "
                       + "r2:" + str(result[6]) + "[Epoch:" + str(result[7]) + "]")

        print("batch_min_rmse:{:.8f}  [Epoch{}_batch{}]".format(self.mim_rmse, self.mim_rmse_epoch, self.mim_rmse_batch))
        print("batch_min_mse:{:.8f}  [Epoch{}_batch{}]".format(self.mim_mse, self.mim_mse_epoch, self.mim_mse_batch))
        print("batch_min_mae:{:.8f}  [Epoch{}_batch{}]".format(self.mim_mae, self.mim_mae_epoch, self.mim_mae_batch))
        print("batch_max_r2:{:.8f}  [Epoch{}_batch{}]".format(self.mim_r2, self.mim_r2_epoch, self.mim_r2_batch))
        print('Min_mean_rmse:{:.8f} [Epoch{}]\t Min_mean_mse:{:.8f} [Epoch{}]\t Min_mean_mae:{:.8f} [Epoch{}]\t Max_mean_r2:{:.8f} [Epoch{}]\t'
              .format(self.min_mean_rmse, self.min_mean_rmse_epoch, self.min_mean_mse, self.min_mean_mse_epoch,
                     self.min_mean_mae, self.min_mean_mae_epoch, self.max_mean_r2, self.max_mean_r2_epoch))

        # plt.figure(figsize=(12, 6))
        # #plt.plot(train_loss['reconstruct'], label='Training Loss')
        # plt.plot(test_loss, label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.savefig('train_valid_loss.png')
        # plt.close()
                
                
            
    def run_async_sampling(self, opt, x1, cond=None, anomaly_mask=None, clip_denoise=False, nfe=None, log_count=20):
        nfe = nfe or opt.interval - 1
        steps = space_indices(opt.interval, nfe + 1)
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        
        device = opt.device
        x1 = x1.to(device)
        cond = cond.to(device) if cond is not None else None
        anomaly_mask = anomaly_mask.to(device) if anomaly_mask is not None else None

        with torch.no_grad():
            with self.ema.average_parameters():
                self.net.eval()

                def pred_x0_fn(xt, step):
                    step = torch.full((xt.shape[0],), step, device=device, dtype=torch.long)
                    out = self.net(xt, step, cond=cond)
                    return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise, anomaly_mask=anomaly_mask)

                xs, pred_x0s = self.diffusion.async_ddpm_sampling(
                    steps=steps,
                    pred_x0_fn=pred_x0_fn,
                    x1=x1,
                    anomaly_mask=anomaly_mask,
                    ot_ode=opt.ot_ode,
                    log_steps=log_steps,
                    verbose=True
                )
                return xs.to(device), pred_x0s.to(device)  



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    opt = EasyDict({
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'use_fp16': False,
        'cond_x1': True,
        'mask_type': 'Cloud_mask',
        'corrup_rate': 68,
        'interval': 1000,
        'beta_max': 0.02,
        'batch_size': 1,
        'microbatch': 1,
        'epoches': 200,
        'lr': 1e-4,
        'l2_norm': 0.01,
        'ema': 0.999,
        'ot_ode': False,
        'clip_denoise': True,
        'visual_epoch': 5,
        'N_S_ratio': 0.1, 
        'save_file': 'South_Sea',
        'path': 'result_0.1_68_test.txt',
        'ckpt_dir': './checkpoints_async_sb',
        't0': 0.0,
        'T': 1.0,
        'tau_min': 0.2,
        'tau_max': 0.5,
        'verbose': True 
    })

    # load data
    x_train, y_train, x_valid, y_valid = deal_sst_util.read_cache(
        f'./data/{opt.N_S_ratio}_{opt.mask_type}_{opt.corrup_rate}_train_{opt.save_file}_miss.h5'
    )
    
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))

   
    # Initialize Runner
    runner = AsyncDSBRunner(opt, log)
    runner.train(opt, train_data, val_data)
