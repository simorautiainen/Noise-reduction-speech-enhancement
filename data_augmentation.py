
import numpy as np
import pickle
import librosa.display
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from math import log10
import matplotlib.pyplot as plt
from sys import float_info
__docformat__ = 'reStructuredText'
__all__ = ['add_noise', 'add_pitch_shift', 'add_impulse_response', 'add_spec_augment']

# return signal with added white gaussian noise of deviation 1 based on the SNR given
def add_noise(input_signal, snr_dB=6):
  # snr is in dbs
  # because when noise is of standard deviation 1 the squared mean of gaussian is 1 so then mean((a*gaussian)^2) =~ a^2
  # looked these formulas from https://en.wikipedia.org/wiki/Signal-to-noise_ratio
  P_signal = np.mean(input_signal**2)

  P_signal_dB = 10*log10(P_signal+float_info.epsilon)

  # derived this formulas on paper
  a = 10**((P_signal_dB-snr_dB)/20)
  noise = a*np.random.normal(size=[len(input_signal)])
  return input_signal+noise

