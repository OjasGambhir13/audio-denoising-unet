import torch
from pystoi import stoi
import numpy as np
from dataset import get_dataloaders
from model import AudioUNet
from tqdm import tqdm

def compute_si_sdr(estimated, target):
    """
    Mathematically computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    This compares the energy of the clean target signal against the error noise.
    """
    eps = 1e-8
    estimated = estimated.view(estimated.size(0), -1)
    target = target.view(target.size(0), -1)
    
    alpha = (torch.sum(estimated * target, dim=-1, keepdim=True) + eps) / (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    
    target_scaled = alpha * target
    noise = estimated - target_scaled
    
    ratio = (torch.sum(target_scaled ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio)

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on: {device}")

    model = AudioUNet().to(device)
    model.load_state_dict(torch.load('models/best_unet.pth', map_location=device, weights_only=True))
    model.eval()

    _, test_loader = get_dataloaders('./datasets/clean', './datasets/noise', batch_size=4)
    
    stoi_scores = []
    sisdr_scores = []

    print("Running mathematical evaluation...")
    with torch.no_grad():
        for noisy_mag, clean_mag in tqdm(test_loader, desc="Testing Audio Batches"):
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)
            
            denoised_mag = model(noisy_mag)
            
            sdr = compute_si_sdr(denoised_mag, clean_mag)
            sisdr_scores.extend(sdr.cpu().numpy().tolist())

            denoised_np = denoised_mag.squeeze().cpu().numpy()
            clean_np = clean_mag.squeeze().cpu().numpy()

            for i in range(denoised_np.shape[0]):
                try:
                    d_audio = denoised_np[i].flatten()
                    c_audio = clean_np[i].flatten()
                    score = stoi(c_audio, d_audio, 16000, extended=False)
                    stoi_scores.append(score)
                except Exception:
                    pass

    print("\n" + "="*40)
    print("🎓 CAPSTONE PROJECT METRICS 🎓")
    print("="*40)
    print(f"Average SI-SDR: {np.mean(sisdr_scores):.2f} dB (Higher is better)")
    print(f"Average STOI:   {np.mean(stoi_scores):.4f} (Max is 1.0)")
    print("="*40)

if __name__ == '__main__':
    evaluate()