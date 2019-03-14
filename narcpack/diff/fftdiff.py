import scipy.fftpack as fft 
import numpy as np

# Based on some good notes I found here:
# https://math.mit.edu/~stevenj/fft-deriv.pdf

def fftdiff(f):
    """Compute the derivative of equally spaced points f using the FFT.
    (Doesn't work so well without periodic input)"""
    N = len(f)
    k = np.zeros(N,dtype=complex)
    if N%2 == 0:
        for i in range(N):
            if i < N/2:
                k[i] = 1j*i
            elif i == N/2:
                k[i] = 0.0
            else:
                k[i] = 1j*(i-N)
    else:
        for i in range(N):
            if i < N/2.0:
                k[i] = 1j*i
            else:
                k[i] = 1j*(i-N)
    return(np.real(fft.ifft(k*fft.fft(f))))
