import numpy as np
from scipy import signal


def smoothed_fd_diff(X, t, method="forward difference", smoother=None, window_length=13):
    """
    Approximate the derivative of set of 1D data points using finite difference methods


    Parameters
    ----------
    X : Numpy 1d or 2d array
        Array of data from which to approximate the derivative. Rows correspond to
        different variables and columns correspond to measurements at different times.

    t : scalar or 1d array
        Times when measurements in X were taken. If scalar, we assume a uniform
        time step.

    method (optional) : str
        Which finite difference method to use. Options include
            "forward difference"
            "centered difference"

    smoother (optional) : str
        Which smoothing method to use. Options include
            "savgol" for a Savitzky-Golay filter
            "median" for a median filter
            "wiener" for a wiener filter

    window_length (optional) : int
        The length of the window the smoother should use.

    """

    # Require an odd window length
    if np.mod(window_length, 2) == 0:
        window_length += 1

    # Apply smoothing
    if smoother:
        if np.ndim(X) == 1:
            X = smooth_data(X, smoother, window_length=window_length).flatten()
        else:
            X = smooth_data(X, smoother, window_length=window_length)

    if method == "forward difference":
        return forward_difference(X, t)
    else:
        return centered_difference(X, t)


# 
# Helper functions
# 

# Function for applying smoothing to data
def smooth_data(X, smoother="savgol", window_length=13):
    # Axis along which to smooth
    axis = np.ndim(X) - 1

    if smoother == "savgol":
        return signal.savgol_filter(X, window_length, polyorder=3, axis=axis)

    if smoother == "median":
        if axis == 0:
            return signal.medfilt(X, kernel_size=window_length)
        else:
            return np.vstack([signal.medfilt(x, kernel_size=window_length) for x in X])

    if smoother == "wiener":
        if axis == 0:
            return signal.wiener(X, mysize=window_length)
        else:
            return np.vstack([signal.wiener(x, mysize=window_length) for x in X])


# First order forward difference
# (and 2nd order backward difference for final point)
def forward_difference(X, t=1, smoother=None, window_length=13):

    # Check whether data is 1D
    if np.ndim(X) == 1:

        # Uniform timestep
        if np.isscalar(t):
            X_diff = (X[1:] - X[:-1]) / t
            backward_diff = np.array([(3*X[-1]/2 - 2*X[-2] + X[-3]/2) / t])
            return np.concatenate((X_diff, backward_diff))

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            X_diff = (X[1:] - X[:-1]) / t_diff
            backward_diff = np.array([(3*X[-1]/2 - 2*X[-2] + X[-3]/2) / t_diff[-1]])
            return np.concatenate((X_diff, backward_diff))

    # Otherwise assume data is 2D
    else:

        # Uniform timestep
        if np.isscalar(t):
            X_diff = (X[:,1:] - X[:,:-1]) / t
            backward_diff = ((3*X[:,-1]/2 - 2*X[:,-2] + X[:,-3]/2) / t).reshape(X.shape[0],1)
            return np.concatenate((X_diff, backward_diff), axis=1)

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            X_diff = (X[:,1:] - X[:,:-1]) / t_diff
            backward_diff = ((3*X[:,-1]/2 - 2*X[:,-2] + X[:,-3]/2) / t_diff[-1]).reshape(X.shape[0],1)
            return np.concatenate((X_diff, backward_diff), axis=1)



# Second order centered difference
# with third order forward/backward difference at endpoints.
# Warning: Sometimes has trouble with nonuniform grid spacing near boundaries
def centered_difference(X, t):
    
    # Check whether data is 1D
    if np.ndim(X) == 1:

        # Uniform timestep
        if np.isscalar(t):
            X_diff = (X[2:] - X[:-2]) / (2 * t)
            forward_diff = np.array([(-11/6 * X[0] + 3 * X[1] \
                - 3/2 * X[2] + X[3] / 3) / t])
            backward_diff = np.array([(11/6 * X[-1]-3 * X[-2] \
                + 3/2 * X[-3]-X[-4]/3) / t])
            return np.concatenate((forward_diff, X_diff, backward_diff))

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            X_diff = (X[2:] - X[:-2]) / (t_diff)
            forward_diff = np.array([(-11/6 * X[0] + 3 * X[1] \
                - 3/2 * X[2] + X[3] / 3) / (t[1]-t[0])])
            backward_diff = np.array([(11/6 * X[-1]-3 * X[-2] \
                + 3/2 * X[-3]-X[-4]/3) / (t[-1]-t[-2])])
            return np.concatenate((forward_diff, X_diff, backward_diff))

    # Otherwise assume data is 2D
    else:

        # Uniform timestep
        if np.isscalar(t):
            X_diff = (X[:,2:] - X[:,:-2]) / (2 * t)
            forward_diff = ((-11/6 * X[:,0] + 3 * X[:,1] \
                - 3/2 * X[:,2] + X[:,3] / 3) / t).reshape(X.shape[0],1)
            backward_diff = ((11/6 * X[:,-1]-3 * X[:,-2] \
                + 3/2 * X[:,-3]-X[:,-4]/3) / t).reshape(X.shape[0],1)
            return np.concatenate((forward_diff, X_diff, backward_diff), axis=1)

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            X_diff = (X[:,2:] - X[:,:-2]) / t_diff
            forward_diff = ((-11/6 * X[:,0] + 3 * X[:,1] \
                - 3/2 * X[:,2] + X[:,3] / 3) / t_diff[0]).reshape(X.shape[0],1)
            backward_diff = ((11/6 * X[:,-1]-3 * X[:,-2] \
                + 3/2 * X[:,-3]-X[:,-4]/3) / t_diff[-1]).reshape(X.shape[0],1)
            return np.concatenate((forward_diff, X_diff, backward_diff), axis=1)
