import numpy as np
import soundfile as sf
import math
from scipy import linalg
from scipy import signal



def get_fft_frame(signal, offset, window, framing_params):

    start_idx = offset
    end_idx = offset + framing_params['frame_length']

    signal_windowed = signal[start_idx:end_idx] * window
    signal_fft_full = np.fft.fft(signal_windowed, n=framing_params['K_fft'], axis=0)
    signal_fft = signal_fft_full[0: framing_params['K_mask']]

    return signal_fft

