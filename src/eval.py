import numpy as np
import torch as th
from torch.utils.data import Subset
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from data.dataset import Dataset
from data.era5 import ERA5
from guided_diffusion.script_util import create_gaussian_diffusion
from model.unet import Unet
from hyperdm import HyperDM
from src.util import parse_test_args


def compute_metrics(predictions, targets):
    metrics = {}
    
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # print("preds np, targets range", preds_np.min(), preds_np.max(), targets_np.min(), targets_np.max())

    # min-max normalize to go from 0-1
    preds_np = (preds_np - preds_np.min()) / (preds_np.max() - preds_np.min() + 1e-8)
    targets_np = (targets_np - targets_np.min()) / (targets_np.max() - targets_np.min() + 1e-8)
    
    psnr_scores = []
    ssim_scores = []
    
    for i in range(len(predictions)):
        pred_img = preds_np[i].squeeze()
        target_img = targets_np[i].squeeze()
        
        # PSNR
        psnr_val = psnr(target_img, pred_img, data_range=1.0)
        psnr_scores.append(psnr_val)
        
        # SSIM
        ssim_val = ssim(target_img, pred_img, data_range=1.0)
        ssim_scores.append(ssim_val)
    
    metrics['psnr_mean'] = np.mean(psnr_scores)
    metrics['psnr_std'] = np.std(psnr_scores)
    metrics['ssim_mean'] = np.mean(ssim_scores)
    metrics['ssim_std'] = np.std(ssim_scores)
    
    return metrics


def evaluate_hyperdm_era5(args):
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    primary_net = Unet(dim=16,
                       dim_mults=(1, 2, 4, 8),
                       channels=1,
                       self_condition=True)
    diffusion = create_gaussian_diffusion(steps=args.diffusion_steps,
                                          predict_xstart=True,
                                          timestep_respacing="ddim25")
    hyperdm = HyperDM(primary_net, args.hyper_net_dims, diffusion).to(device)
    hyperdm.load_state_dict(th.load(args.checkpoint, weights_only=True))
    hyperdm.eval()
    
    dataset = ERA5(args.image_size, split="test", download=False)
    test_size = 50 # artificially low, can raise if more compute
    dataset = Subset(dataset, range(test_size))
    
    print(f"Evaluating on {test_size} test samples...")
    
    all_preds = []
    all_targets = []
    all_epistemic = []
    all_aleatoric = []
    
    for idx in tqdm(range(test_size)):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        
        with th.no_grad():
            mean, var = hyperdm.get_mean_variance(
                M=args.M,
                N=args.N,
                condition=y,
                device=device,
                progress=False
            )
        
        pred = mean.mean(dim=0)
        epistemic = mean.var(dim=0)
        aleatoric = var.mean(dim=0)
        
        all_preds.append(pred)
        all_targets.append(x.cpu())
        all_epistemic.append(epistemic)
        all_aleatoric.append(aleatoric)
    
    all_preds = th.stack(all_preds)
    all_targets = th.cat(all_targets)
    all_epistemic = th.stack(all_epistemic)
    all_aleatoric = th.stack(all_aleatoric)
    total_uncertainty = all_epistemic + all_aleatoric
    
    print("\n=== Evaluation Results ===")
    
    basic_metrics = compute_metrics(all_preds, all_targets)
    print(f"PSNR: {basic_metrics['psnr_mean']:.3f} ± {basic_metrics['psnr_std']:.3f}")
    print(f"SSIM: {basic_metrics['ssim_mean']:.4f} ± {basic_metrics['ssim_std']:.4f}")
    
    print(f"\nEpistemic Uncertainty: {all_epistemic.mean():.6f}")
    print(f"Aleatoric Uncertainty: {all_aleatoric.mean():.6f}")
    print(f"Total Uncertainty: {total_uncertainty.mean():.6f}")
    
    return {**basic_metrics}


if __name__ == "__main__":
    args = parse_test_args()
    print(args)
    
    if args.dataset == Dataset.ERA5:
        metrics = evaluate_hyperdm_era5(args)
    # elif args.dataset == Dataset.TOY:
    #     metrics = evaluate_hyperdm_toy(args)
    else:
        raise NotImplementedError()
    
    import json
    with open(f"{args.dataset}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {args.dataset}_metrics.json")