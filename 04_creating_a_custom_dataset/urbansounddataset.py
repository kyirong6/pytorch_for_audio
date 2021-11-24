"""Reference series: https://www.youtube.com/watch?v=88FFnqt5MNI&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm&index=4
"""
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os


class UrbanSoundDataset(Dataset):
    
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        
    
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
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    
    
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
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
    
    print(f"There are {len(usd)} samples in the dataset.")
    
    signal, label = usd[0]
    
    print(signal, label)
    