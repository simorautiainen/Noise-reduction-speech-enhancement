#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union, MutableMapping
import os
import pathlib
import itertools
import numpy as np
import matplotlib.pyplot as plt
from librosa.core import load as lb_load, stft
from librosa.filters import mel
from typing import MutableSequence
from typing import Optional
import pickle
import os
from scipy.signal import resample
import librosa

__docformat__ = 'reStructuredText'
__all__ = [ 'create_one_hot_encoding',
            'get_audio_file_data',
            'to_audio',
            'serialize_features_and_classes'
           ]


def create_one_hot_encoding(word: str,
                            unique_words: MutableSequence[str]) \
        -> np.ndarray:
    """Creates an one-hot encoding of the `word` word, based on the\
    list of unique words `unique_words`.

    :param word: Word to generate one-hot encoding for.
    :type word: str
    :param unique_words: List of unique words.
    :type unique_words: list[str]
    :return: One-hot encoding of the specified word.
    :rtype: numpy.ndarray
    """
    to_return = np.zeros((len(unique_words)))
    to_return[unique_words.index(word)] = 1
    return to_return


def get_audio_file_data(audio_file: Union[str, pathlib.Path], sr: Optional[int] = None) \
        -> np.ndarray:
    """Loads and returns the audio data from the `audio_file` with a specific sr by resampling it.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str or pathlib.Path
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    return lb_load(path=audio_file, sr=sr, mono=True)

def to_audio(mix_waveform: np.ndarray, 
             predicted_vectors: np.ndarray,
             n_ffts: int = 1024,
             win_length: int = 512,
             hop_size: int = 256,
             win_type: str = 'hann') \
    -> np.ndarray:
	"""
	:param mix_waveform: The waveform of the monaural mixture. Expected shape (n_samples,)
    :type mix_waveform: numpy.ndarray
	:param predicted_vectors: A numpy array of shape: (chunks, frequency_bins, time_frames)
    :type predicted_vectors: numpy.ndarray
	:return: predicted_waveform: The waveform of the predicted signal: (~n_samples,)
    :rtype: numpy.ndarray
	"""
	# Pre-defined (I)STFT parameters

	# STFT analysis of waveform
	c_x = librosa.stft(mix_waveform, n_fft=n_ffts, hop_length=hop_size, win_length=win_length, window=win_type)[:512,:]
	# Phase computation
	phs_x = np.angle(c_x)
	# Get the number of time-frames
	tf = phs_x.shape[1]

	# Number of chunks/sequences
	n_chunks, fb, seq_len = predicted_vectors.shape
	p_end = seq_len*n_chunks
	# Reshaping
	rs_vectors = np.reshape(np.moveaxis(predicted_vectors, 0, 1), (fb, p_end))
	# Reconstruction
	if p_end > tf:
		# Appending zeros to phase
		c_vectors = np.hstack((phs_x, np.zeros_like(phs_x[:, :p_end-seq_len])))
	else:
		c_vectors = rs_vectors * np.exp(1j * phs_x[:, :p_end])
	# ISTFT
	predicted_waveform = librosa.istft(c_vectors, n_fft=n_ffts, hop_length=hop_size, win_length=win_length, window=win_type)


	return predicted_waveform

def serialize_features_and_classes(parent_folder_path: pathlib.Path, file_name: str, features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) -> None:
    """Serializes the features and classes.

    :file_name: file name without extension to which save
    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    if not parent_folder_path.exists():
        parent_folder_path.mkdir(parents=True)
    with open(parent_folder_path.joinpath(file_name), "wb") as f:
        pickle.dump(features_and_classes, f)

def extract_mel_band_energies(audio_file: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              hop_length: Optional[int] = 512,
                              win_length: Optional[int] = 1024,
                              n_mels: Optional[int] = 40) \
        -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :param sr: Sampling frequency of audio file, defaults to 44100.
    :type sr: Optional[int]
    :param n_fft: STFT window length (in samples), defaults to 1024.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 512.
    :type hop_length: Optional[int]
    :param n_mels: Number of MEL frequencies/filters to be used, defaults to 40.
    :type n_mels: Optional[int]
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    spec = stft(
        y=audio_file,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length)

    mel_filters = mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    return np.dot(mel_filters, np.abs(spec) ** 2)

# EOF
