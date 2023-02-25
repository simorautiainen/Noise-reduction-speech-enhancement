from utils import get_audio_file_data, serialize_features_and_classes
import csv
from typing import List
import pathlib
import numpy as np
from data_augmentation import add_noise
import random
import librosa
import sounddevice as sd
import time

SR = 16000
SEGMENT_SECONDS = 1
SEGMENT = round(SEGMENT_SECONDS*SR)

def get_file_path(paths: List[pathlib.Path], file_name: str) -> pathlib.Path:
    for path in paths:
        if path.name == file_name:
            return path
    return pathlib.Path()


def create_features(audio_folder: str, out_folder:str, stop_at_n_file = 15000):
    paths = pathlib.Path(audio_folder)
    amount_of_legit_files = 0
    for i, file in enumerate(paths.rglob("*")):
        if file.suffix != ".flac":
            continue
        # I have so many files so I need to stop generating the features at some point
        if amount_of_legit_files > stop_at_n_file:
            break
        amount_of_legit_files+=1
        print(f"currently {amount_of_legit_files}th file named {str(file)}")
        data, sr = get_audio_file_data(file, SR)
        segments = []

        while data.size > SEGMENT:
            segments.append(data[:SEGMENT])
            data = data[SEGMENT:]
        for k, a_segment in enumerate(segments):
            noise_segment = add_noise(a_segment, 6)
        
            stft = librosa.stft(a_segment, n_fft=1024, hop_length=256, win_length=512)
            stft_noise = librosa.stft(noise_segment, n_fft=1024, hop_length=256, win_length=512)
            features = {"features": stft_noise, "target": stft}
            parentPath = pathlib.Path(out_folder, file.stem)
            nmber = str(k+1).zfill(4)
            serialize_features_and_classes(parentPath,f"{nmber}-segment.pickle", features)
            


if __name__ == "__main__":
    max_files = 150000
    create_features("LibriSpeech", "features/train", max_files)
    create_features("LibriSpeechTest", "features/test", max_files)
    # data, sr = get_audio_file_data("61-70968-0000.flac", SR)
    # noise_data = add_noise(data, 6)
    # sd.play(noise_data, sr)
    # time.sleep(5)
    # sd.stop()