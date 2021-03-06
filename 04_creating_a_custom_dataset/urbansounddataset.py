"""Reference series: https://www.youtube.com/watch?v=88FFnqt5MNI&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=4
"""
import torch
from torch._C import _set_worker_signal_handlers
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, 
                 target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
        
    def __len__(self):
        """Creates a custom way of calculating len(usd). This is a magic method.
        """
        return len(self.annotations)
    
    
    def __getitem__(self, index):
        """Allows our UrbanSoundDataset class to be indexable. This is a magic method. For example:
        
            usd_instance[1]

        Args:
            index: an int
        """
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # signal -> (num_channels, sample) -> (2, 16000) : this means it is stereo because of 2 channels
        # ideally: signal -> (1, 16000) : 1 channel
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (num_channels, num_samples) -> (1, 50000) -> (1, 22050)
        if signal.shape[1] > self.num_samples:
            # only keep up to 22050 (self.num_samples)
            signal = signal[:, :self.num_samples] 
        return signal
            
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # right pad: [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    
    def _resample_if_necessary(self, signal, sr):
        """Resample the signel at a target sample rate so it is standard across all signals.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    
    def _mix_down_if_necessary(self, signal):
        """If a signal has multiple channel we would like to mix it down to a 
        single channel.
        """
        # for ex: (2, 1000), num of channels is first value
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
        
    
    def _get_audio_sample_path(self, index):
        # creating path to the audio file
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    

if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/choendenkyirong/Desktop/developer/code/src/github/kyirong6/pytorch_for_audio/datasets/UrbanSound8K/metadata/UrbanSound8k.csv"
    AUDIO_DIR = "/Users/choendenkyirong/Desktop/developer/code/src/github/kyirong6/pytorch_for_audio/datasets/UrbanSound8K/audio" 
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using device: {device}")
    
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        )
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
   #print(signal, label)