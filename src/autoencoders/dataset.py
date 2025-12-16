from torch.utils.data import Dataset
import torchaudio
import glob

class WavDataset(Dataset):
    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = sorted(glob.glob(audio_dir + "/*.wav"))
        if not self.filenames:
            raise RuntimeError(f"No wav files found in {audio_dir}")
        
        x, self.sr = torchaudio.load(self.filenames[0])
        self.cc = x.size(0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        x, sr = torchaudio.load(self.filenames[index])  # [C, T]

        # safety checks
        assert sr == self.sr, f"Unexpected sampling rate: {sr} != {self.sr}"
        assert x.size(0) == self.cc, f"Unexpected channel count: {x.size(0)} != {self.cc}"

        return x
