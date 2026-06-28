import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, n_fft=512, hop_length=256):
        
        self.clean_files = glob.glob(os.path.join(clean_dir, '**/*.flac'), recursive=True) + \
                           glob.glob(os.path.join(clean_dir, '**/*.wav'), recursive=True)
                           
        
        self.noise_files = glob.glob(os.path.join(noise_dir, '**/*.wav'), recursive=True) + \
                           glob.glob(os.path.join(noise_dir, '**/*.flac'), recursive=True)
        
        
        if len(self.clean_files) == 0:
            raise ValueError(f"No audio files found in the clean directory: {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No audio files found in the noise directory: {noise_dir}")

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.window = torch.hann_window(self.n_fft)

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        
        clean_path = self.clean_files[idx]
        clean_audio, sr = torchaudio.load(clean_path)
        
        
        if clean_audio.shape[0] > 1: 
            clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
        
        
        target_len = sr * 3
        if clean_audio.shape[1] > target_len:
            clean_audio = clean_audio[:, :target_len]
        else:
            pad_amount = target_len - clean_audio.shape[1]
            clean_audio = torch.nn.functional.pad(clean_audio, (0, pad_amount))

        
        noise_path = random.choice(self.noise_files)
        noise_audio, _ = torchaudio.load(noise_path)
        
        
        if noise_audio.shape[0] > 1: 
            noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)
        
       
        if noise_audio.shape[1] < target_len:
            repeat_factor = (target_len // noise_audio.shape[1]) + 1
            noise_audio = noise_audio.repeat(1, repeat_factor)
        noise_audio = noise_audio[:, :target_len]

        
        noisy_audio = clean_audio + 0.3 * noise_audio

        
        clean_stft = torch.stft(clean_audio.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        noisy_stft = torch.stft(noisy_audio.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)

        
        clean_mag = torch.abs(clean_stft).unsqueeze(0)[:, :256, :]
        noisy_mag = torch.abs(noisy_stft).unsqueeze(0)[:, :256, :]

        return noisy_mag, clean_mag
def get_dataloaders(clean_dir, noise_dir, batch_size=8):
    dataset = AudioDataset(clean_dir, noise_dir)
    
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

   
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            drop_last=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, val_loader