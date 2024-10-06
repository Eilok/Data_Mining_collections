def get_targets(train_solution):
    '''
    Returns the targets for wc and atmosphere
    '''
    targets = train_solution[:, 1:]
    targets_mean = targets[:, 1:].mean(axis=1) # get rid of FGS data
    return targets, targets_mean

def get_targets_wc(targets_mean):
    '''
    returns normalized targets for wc
    '''
    max_targets = targets_mean.max()
    min_targets = targets_mean.min()
    targets_norm = (targets_mean - min_targets)/(max_targets - min_targets)
    return targets_norm, min_targets, max_targets

def suppress_mean(targets, mean): 
    '''
    leave the atmospheric features only
    '''
    res = targets - np.repeat(mean.reshape((mean.shape[0], 1)), repeats = targets.shape[1], axis = 1)
    return res

def targets_normalization(targets_shift):
    '''
    normalize targets (max abs normalization)
    '''
    max_targets = targets_shift.max()
    min_targets = targets_shift.min()
    targets_abs_max = np.max([min_targets, max_targets])
    targets_norm = targets_shift/targets_abs_max
    return targets_norm, targets_abs_max

def targets_norm_back (data, data_abs_max) : 
    return data * data_abs_max