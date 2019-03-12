import scipy.fftpack as fft 
import numpy as np

def fftdiff(x):
    return(fft.ifft(2j*np.pi*x*fft.fft(x)))
