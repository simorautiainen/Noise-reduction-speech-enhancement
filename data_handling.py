
from typing import Tuple, Optional, Union, Dict, List
from pickle import load as pickle_load
from pathlib import Path
import sounddevice as sd
from torch.utils.data import Dataset
import numpy as np
import librosa
import pickle
from utils import to_audio
import time
import librosa.display as display
import matplotlib.pyplot as plt
__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']

def get_items_from_dir_with_pathlib(dir_name: Union[str, Path]) \
        -> List[Path]:
    """Returns the items in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(Path(dir_name).iterdir())

class MyDataset(Dataset):

    def __init__(self,
                direc: Union[str, Path],
                full_audio: Optional[bool] = False) \
            -> None:
        super().__init__()

        # If pathlib sorts these arrays then everything should work fine
        self.the_dir = Path(direc)
        self.full_audio = full_audio
        if self.full_audio:
            self.files = get_items_from_dir_with_pathlib(self.the_dir)
        else:
            self.files = [*filter(lambda file: file.suffix==".pickle",self.the_dir.rglob("*"))]
    @staticmethod
    def _full_stft_to_audio(stft_splitted: np.ndarray):
        return librosa.istft(np.concatenate(stft_splitted, axis=1), n_fft=1024, win_length=512, hop_length=256, window='hann')
    
    @staticmethod
    def _load_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The files content containing the noiseless audio and noise audio.
        :rtype: dict[str, int|numpy.ndarray]
        """
        with file_path.open('rb') as f:
            features = pickle.load(f)
            return features["features"], features["target"]
    @staticmethod
    def _load_all_files_in_folder(folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:

        all_files = get_items_from_dir_with_pathlib(folder_path)
        correct_sorting = lambda x : (len(str(x)), x)
        sorted_files = sorted(all_files, key=correct_sorting)
        list_of_features = []
        list_of_targets = []
        for file in sorted_files:
            with file.open('rb') as f:
                features = pickle.load(f)
                list_of_features.append(features["features"])
                list_of_targets.append(features["target"])
        return np.array(list_of_features), np.array(list_of_targets)
    
    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, int)
        """
        if self.full_audio:
            features, target = self._load_all_files_in_folder(self.files[item])
            features = features[:, :512, :]
            target = target[:, :512, :]
            full_song = self._full_stft_to_audio(features)
            return features, target, full_song
        else:
            features, target = self._load_file(self.files[item])
            return np.abs(features)[:512, :], np.abs(target)[:512, :]




if __name__ == "__main__":
    # train_dataset = MyDataset("features/train")
    # print(len(train_dataset))
    # features, target = train_dataset[1]
    
    # fig, [ax1 , ax2] = plt.subplots(1, 2, figsize=[5, 10])
    # librosa.display.specshow(np.log10(features**2), ax=ax1, vmin=-5, vmax=2)
    # img = librosa.display.specshow(np.log10(target**2), ax=ax2, vmin=-5,vmax=2)

    
    # plt.colorbar(img)
    # plt.show()

    # print(features.shape)
    test_dataset = MyDataset("features/test", True)
    features, target, full_song = test_dataset[5]
    print("target shape", target.shape)
    audio = to_audio(full_song, np.abs(target))
    sd.play(audio, 16000)
    time.sleep(5)
    sd.stop()
