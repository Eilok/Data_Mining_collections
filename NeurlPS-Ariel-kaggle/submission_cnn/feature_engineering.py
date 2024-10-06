def ADC_convert(signal, gain, offset):
    """
    Converts ADC (Analog-to-Digital Converter) signals to actual values.

    This function is used to convert digital signals obtained from an ADC to their corresponding physical values based on the gain and offset.

    Parameters:
    - signal: Array of digital signals obtained from the ADC.
    - gain: Gain value used to scale the signal intensity back to its actual value.
    - offset: Offset value used to adjust the signal to the correct measurement range.

    Returns:
    - Corrected signal array representing the actual physical values.
    """
    # Convert the signal to floating-point numbers to improve precision
    signal = signal.astype(np.float64)
    
    # Divide the signal by the gain to restore the signal intensity
    signal /= gain
    
    # Add the offset to adjust the signal to the correct measurement range
    signal += offset
    
    # Return the corrected signal
    return signal

def readout_correct(signal, read_noise):
    """
    Subtract readout noise from the signal.

    Parameters:
    - signal: Signal array from which readout noise is to be subtracted.
    - read_noise: Readout noise array to be subtracted from the signal array.

    Returns:
    - signal_readout: Signal array after subtracting readout noise.
    """
    # Tile the readout noise array to match the dimensions of the signal array.
    read_noise = np.tile(read_noise, (signal.shape[0], 1, 1))
    # Subtract readout noise from the signal.
    signal_readout = signal - read_noise
    return signal_readout

def mask_hot_dead(signal, dead, dark, maxiters=5, sigma=5):
    '''
    This function identifies and masks hot and dead pixels in the input signal.

    Parameters:
    - signal: The primary input data, typically raw signals collected from astronomical instruments (like images or time series data). The shape of the signal matrix is usually (time, height, width), representing images at different times (or multiple frames).

    - dead: A matrix indicating dead pixels. This matrix marks which pixels in the sensor are dead (i.e., do not respond to light signals). The shape of the dead matrix should match a single frame image, typically (height, width).

    - dark: Represents "dark frame" data. Dark frames are measured under no-light conditions to capture hot pixels or other electronic noise in the detector. The shape of the dark matrix also matches a single frame image.
    
    Returns:
    - signal: The processed signal with hot and dead pixels masked.
    '''
    # Initialize hot pixel mask as a boolean array of False (no pixels masked initially)
    hot_mask = np.zeros_like(dark, dtype=bool)
    
    # Iteratively mask values that exceed the sigma threshold
    for _ in range(maxiters):
        # Calculate mean and standard deviation of unmasked values
        mean_dark = np.mean(dark[~hot_mask])
        std_dark = np.std(dark[~hot_mask])
        
        # Find new hot pixels exceeding the sigma threshold
        new_mask = np.abs(dark - mean_dark) > sigma * std_dark
        
        # Update hot_mask to include newly detected hot pixels
        hot_mask |= new_mask  # Use bitwise OR to accumulate masks
        
        # If no new pixels are being masked, stop iterating
        if not np.any(new_mask):
            break
    # Extend the hot and dead pixel masks across the entire time dimension to match the shape of the signal
    hot = np.tile(hot_mask, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    # Mask dead and hot pixels
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal

def apply_linear_corr(linear_corr, clean_signal):
    """
    Apply linear correction to the clean signal.

    Parameters:
    linear_corr: Array of polynomial correction coefficients, where each row corresponds to a polynomial's coefficients.
    clean_signal: Original signal data, a three-dimensional array where the first dimension represents the signal sequence, and the last two dimensions represent the image's two-dimensional pixel coordinates.

    Returns:
    Corrected signal.
    """
    # Reverse the linear_corr array along the first dimension
    # This is because the coefficients in np.poly1d are ordered from highest degree to lowest, while the linear_corr array stores them in reverse order
    linear_corr = np.flip(linear_corr, axis=0)
    
    # Generate coordinate pairs (x, y) for each pixel in the image and iterate through each pixel
    # This is to apply polynomial correction to each pixel individually
    for x, y in itertools.product(
                range(clean_signal.shape[1]), range(clean_signal.shape[2])
            ):
        # For each pixel (x, y), extract the corresponding polynomial coefficients linear_corr[:, x, y], and construct a polynomial function poli using np.poly1d
        # This converts the polynomial coefficients into an evaluable polynomial function
        poli = np.poly1d(linear_corr[:, x, y])
        
        # Apply the polynomial poli to the original signal clean_signal[:, x, y] and replace the original signal with the corrected result
        # This directly replaces the original signal data with the output of the polynomial function, achieving the correction
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    
    # Return the corrected signal
    return clean_signal

def clean_dark(signal, dead, dark, dt):
    """
    Subtract dark current from the given signal, handling dead pixels.

    Parameters:
    signal: ndarray, observed signal to be corrected for dark current.
    dead: ndarray, mask array indicating dead pixels, used to mask out dead pixels in dark current data.
    dark: ndarray, dark current data, used to correct the signal.
    dt: ndarray, array of time differences, used to adjust the contribution of dark current.

    Returns:
    ndarray, corrected signal with dark current removed.
    """
    # Mask out dead pixels in the dark current data
    dark = np.ma.masked_where(dead, dark)
    # Tile the dark data along the time axis to match the shape of the signal.
    dark = np.tile(dark, (signal.shape[0], 1, 1))
    # Subtract the dark current from the observed signal.
    signal -= dark * dt[:, np.newaxis, np.newaxis]  # Extend the shape of dt to enable broadcasting across the spatial dimensions of the signal.
    return signal

def get_cds(signal):
    """
    Calculate and return the differential signal (CDS).

    The CDS is computed by taking the difference between the odd and even columns of the input signal.
    This method is commonly used in image processing and signal processing to capture local changes in the signal.

    Parameters:
    signal (ndarray): A four-dimensional array representing the input signal. The second dimension is interpreted as rows,
                      where odd and even columns are subtracted to compute the differential signal.

    Returns:
    ndarray: An array containing the computed differential signal (CDS). The returned array has the same dimensions as the
             input signal, except that the second dimension is halved because each pair of columns is combined to compute a single difference.
    """
    # Compute the differential signal by subtracting even columns from odd columns
    cds = signal[:,1::2,:,:] - signal[:,::2,:,:]
    return cds

def correct_flat_field(flat, dead, signal):
    '''
    Function to perform flat field correction.
    Flat field correction aims to correct the non-uniform response of each pixel in the detector, improving the accuracy of the observed signal.
    
    Parameters:
    flat: Flat field image representing the response differences of each pixel in the detector.
    dead: Mask for dead pixels, identifying failed or abnormal pixels (e.g., pixels that do not respond normally).
    signal: Original observed signal that needs correction.
    
    Returns:
    Corrected observed signal.
    '''
    # Transpose the flat field and dead pixel images for subsequent processing
    # flat = flat.transpose(1, 0)
    # dead = dead.transpose(1, 0)
    
    # Mask the flat field image to ignore the effect of dead pixels
    flat = np.ma.masked_where(dead, flat)
    
    # Expand the flat field image along the time dimension to match the signal dimensions
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    
    # Perform flat field correction to remove non-uniform pixel responses
    signal = signal / flat
    # The flat field contains the response coefficients of each pixel, and this step eliminates the non-uniform response of different pixels, resulting in a more accurate signal.
    
    return signal

def bin_obs(signal, binning):
    '''
    Performs time binning on the signal data processed by Correlated Double Sampling (CDS).

    Parameters:
    cds_signal: The input signal data after Correlated Double Sampling, with shape (time, height, width).
    binning: The stride for time binning, indicating how many time steps are merged into one.

    Returns:
    Binned signal data with reduced time steps and unchanged dimensions.
    '''
    # height*width -> width*height
    signal_transposed = signal.transpose(0,1,3,2)
    
    # Create a new empty array `cds_binned` to store the binned signal data,
    # reducing the time dimension length by a factor of 1/binning since multiple time steps are merged.
    # Other dimensions remain unchanged.
    signal_binned = np.zeros((signal_transposed.shape[0], signal_transposed.shape[1]//binning, signal_transposed.shape[2], signal_transposed.shape[3]))
    
    # Iterate over each new time step `i`
    for i in range(signal_transposed.shape[1]//binning):
        # Select `binning` consecutive time steps using i*binning:(i+1)*binning and merge them.
        # np.sum(..., axis=1) sums these time steps along the time dimension, achieving time binning.
        signal_binned[:,i,:,:] = np.sum(signal_transposed[:,i*binning:(i+1)*binning,:,:], axis=1)
    
    # Return the binned signal data with reduced time steps and unchanged dimensions
    return signal_binned