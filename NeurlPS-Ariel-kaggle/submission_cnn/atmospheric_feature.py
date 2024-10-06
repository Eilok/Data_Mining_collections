def combine_data(AIRS, FGS):
    FGS_column = FGS.sum(axis=2)
    dataset = np.concatenate([AIRS, FGS_column[:, :, np.newaxis, :]], axis=2)
    dataset = dataset.sum(axis=3)
    return dataset

def norm_star_spectrum (signal):
    '''
    normalize the signal by star spectrum
    ''' 
    # extract the mean spectrum of the star
    img_star = signal[:,:50].mean(axis = 1) + signal[:,-50:].mean(axis = 1)
    # signal is divided by mean star spectrum to avoid the influence of star
    return signal/img_star[:,np.newaxis,:]

def suppress_out_transit(data, ingress, egress):
    '''
    only consider the process of in transit
    ''' 
    data_in = data[:, ingress:egress,:]
    return data_in

def substract_data_mean(data):
    '''
    remove the spectrum from basic light and leave the atmospheric feature only
    '''
    data_mean = np.zeros(data.shape)
    for i in range(data.shape[0]):
        data_mean[i] = data[i] - data[i].mean()
    return data_mean

def data_norm(data):
    data_min = data.min()
    data_max = data.max()
    data_abs_max = np.max([data_min, data_max])
    data_norm = data/data_abs_max
    return data_norm, data_abs_max

def data_normback(data, data_abs_max) : 
    return data * data_abs_max