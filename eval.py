import torch
from data_handling import MyDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import soundfile as sf
import numpy as np
import pathlib
from mir_eval.separation import bss_eval_images_framewise as bss
from model.Net import Net
from enum import Enum
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from typing import Tuple
from utils import get_audio_file_data, to_audio, extract_mel_band_energies
from speechbrain.pretrained import SepformerSeparation
from dcase_util.containers import AudioContainer
from time import time
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sys import float_info
def mir(ground_truth, predicted):
    (sdr, isr, sir, sar, perm) = bss(ground_truth, predicted, window=44100*1, hop=1*44100)
    return sir, sar, sdr
def transform_and_evaluate(test_loader: DataLoader, state_loc: str = "savedstates/20epochs15000samples/state.pt", samples=3, sr=16000):
    net = Net()
    net = net.to(dtype=torch.float)
    net.load_state_dict(torch.load(state_loc))
    net.eval()
    testing_loss = []
    device = "cpu"
    loss_function = MSELoss()
    start_time = time()
    print("MY MODEL:")
    with torch.no_grad():
        for i, val in zip(range(samples), test_loader):
            mixture, source, full_song = val
            mixture = torch.squeeze(mixture)
            source = torch.squeeze(source)
            full_song = torch.squeeze(full_song)
            mixture = mixture.to(device)
            source = source.to(device)

            source_hat = net(torch.abs(mixture))
            loss = loss_function(input=source_hat, target=torch.abs(source).type_as(source_hat))

            testing_loss.append(loss.item())
            audio = to_audio(full_song.numpy(), source_hat.numpy())
            p = pathlib.Path(f"sourcehats/{i}")
            if not p.exists():
                p.mkdir(parents=True)
            sf.write(f'{str(p)}/estimated_from_noisy.flac', audio, sr, format='flac')
            full_song_source = MyDataset._full_stft_to_audio(source.numpy())
            sf.write(f'{str(p)}/actual.flac', full_song_source, sr, format='flac')
            sf.write(f'{str(p)}/with_noise.flac', full_song.numpy(), sr, format='flac')
            print("\tfile", i+1, "done, currently taken time", time()-start_time)
    print(f"Took my model {time()-start_time}s for to handle {samples} samples")
    testing_loss = np.array(testing_loss).mean()
    print(testing_loss)

def do_speechbrain_estimation(samples=3, sr=16000):
    speech_enhancer = SepformerSeparation.from_hparams(
        source='speechbrain/sepformer-whamr16k', 
        savedir='pretrained_models/sepformer-whamr16k'
    )
    start_time = time()
    print("SPEECHBRAIN:")
    for i in range(samples):

        parent = pathlib.Path("sourcehats", str(i))
        
        noisy_mixture = AudioContainer().load(
            filename=str(pathlib.Path(parent, "with_noise.flac")), 
            fs=sr, 
            mono=True
        )
        enhanced = speech_enhancer.separate_batch(torch.from_numpy(noisy_mixture.data).float()[None,:])
        enhanced = enhanced.mean(2).squeeze().numpy() # outputs two channels
        enhanced = enhanced/max(enhanced.max(), enhanced.min(), key=abs)
        sf.write(f"{str(parent)}/estimated_from_noisy_speechbrain.flac", enhanced, sr, format="flac")
        print("\tfile", i+1, "done, currently taken time", time()-start_time)
    print(f"Took speechbrain {time()-start_time}s for to handle {samples} samples")

def calc_pesq(predicted: np.ndarray, target: np.ndarray, fs: int = 16000 ) -> Tuple[float, float]:
    """Returns tuple containing (narrow_band, wide_band) perceptual evaluation speech quality metrics for given audio.

    :param predicted: The predicted audio.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    predicted_tensor = torch.Tensor(predicted)
    target_tensor = torch.Tensor(target)
    narrow_band = perceptual_evaluation_speech_quality(predicted_tensor, target_tensor, fs, 'nb')
    wide_band = perceptual_evaluation_speech_quality(predicted_tensor, target_tensor, fs, 'wb')
    return narrow_band.item(), wide_band.item()

def save_mel_spectrograms(directory: str = "sourcehats", sr:int =16000, hop_length=512, win_length=1024, n_mels=40):
    for folder in pathlib.Path(directory).iterdir():
        clean_audio,_ = librosa.load(str(pathlib.Path(folder, "actual.flac")), mono=True, sr=sr)
        noisy_audio,_ = librosa.load(str(pathlib.Path(folder, "with_noise.flac")), mono=True, sr=sr)
        estimated_audio,_ = librosa.load(str(pathlib.Path(folder, "estimated_from_noisy.flac")), mono=True, sr=sr)
        estimated_speechbrain,_ = librosa.load(str(pathlib.Path(folder, "estimated_from_noisy_speechbrain.flac")), mono=True, sr=sr)

        clean_audio_mel_stft = extract_mel_band_energies(clean_audio, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        noisy_mel_stft = extract_mel_band_energies(noisy_audio, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        estimated_audio_mel_stft = extract_mel_band_energies(estimated_audio, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        estimated_speechbrain_mel_stft = extract_mel_band_energies(estimated_speechbrain, n_fft=1024, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
        
        log_power_clean = np.log10(clean_audio_mel_stft + float_info.epsilon)
        log_power_noisy = np.log10(noisy_mel_stft + float_info.epsilon)

        log_power_estimated = np.log10(estimated_audio_mel_stft + float_info.epsilon)
        log_power_estimated_speechbrain = np.log10(estimated_speechbrain_mel_stft + float_info.epsilon)

        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=[15, 15])
        librosa.display.specshow(log_power_clean, ax=ax1, y_axis='mel', x_axis='time', hop_length=hop_length, vmin=-5, vmax=5)
        librosa.display.specshow(log_power_noisy, ax=ax2, y_axis='mel', x_axis='time', hop_length=hop_length, vmin=-5, vmax=5)
        librosa.display.specshow(log_power_estimated, ax=ax3, y_axis='mel', x_axis='time', hop_length=hop_length, vmin=-5, vmax=5)
        img = librosa.display.specshow(log_power_estimated_speechbrain, ax=ax4, y_axis='mel', x_axis='time', hop_length=hop_length, vmin=-5, vmax=5)

        ax1.set_title("Clean audio")
        ax2.set_title("Noisy audio")
        ax3.set_title("Estimated project model")
        ax4.set_title("Estimated speechbrain sepformer")
        cax = plt.axes([0.95, 0.1, 0.01, 0.8])
        fig.colorbar(img, cax=cax)
        fig.savefig(str(pathlib.Path(folder, "spectrograms.jpg")), dpi=300)

if __name__ == "__main__":
    test_loader = DataLoader(MyDataset("features/test", True), batch_size=1, shuffle=True)
    sr = 16000
    samples = 20
    transform_and_evaluate(test_loader, samples=samples, sr=sr) # generates the source hats

    do_speechbrain_estimation(samples, sr) # does the speechbrain estimation on the audios

    my_model_scores = {}
    pesq_scores_df = pd.DataFrame(columns=["my_model_narrow_band", "my_model_wide_band", "speechbrain_narrow_band", "speechbrain_wide_band"], dtype=np.float32)
    for i in range(samples):
        parent = pathlib.Path("sourcehats", str(i))
        target, _ = get_audio_file_data(pathlib.Path(parent, "actual.flac"), sr)
        predicted, _ = get_audio_file_data(pathlib.Path(parent, "estimated_from_noisy.flac"), sr)
        predicted_using_speech_brain, _ = get_audio_file_data(pathlib.Path(parent, "estimated_from_noisy_speechbrain.flac"), sr)

        target_normalized = target/max(target.max(), target.min(), key=abs)
        predicted_normalized  = predicted/max(predicted.max(), predicted.min(), key=abs)
        predicted_using_speech_brain_normalized  = predicted_using_speech_brain/max(predicted_using_speech_brain.max(), predicted_using_speech_brain.min(), key=abs)
        
        my_model_pesq = calc_pesq(predicted_normalized, target_normalized, sr)
        speechbrain_pesq = calc_pesq(predicted_using_speech_brain_normalized, target_normalized, sr)
        pesq_scores_df.loc[i] = [*my_model_pesq, *speechbrain_pesq]
        print("My model on sample {}: Pesq narrow band: {}, pesq wide band: {}".format(i, *my_model_pesq))
        print("Speech brain on sample {} : Pesq narrow band: {}, pesq wide band: {}\n".format(i, *speechbrain_pesq))
    pesq_scores_df.to_csv('pesq_scores.csv')
    save_mel_spectrograms()