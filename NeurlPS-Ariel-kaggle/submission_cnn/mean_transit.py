def get_wc(AIRS):
    '''
    get the normalized white curve
    '''
    AIRS = AIRS.sum(axis=3)
    wc_mean = AIRS.mean(axis=1).mean(axis=1)
    white_curve = AIRS.sum(axis=2) / wc_mean[:, np.newaxis]
    return white_curve

def normalize_wlc(wlc):
    wlc_min = wlc.min()
    wlc_max = wlc.max()
    wlc_norm = (wlc - wlc_min) / (wlc_max - wlc_min)
    return wlc_norm

