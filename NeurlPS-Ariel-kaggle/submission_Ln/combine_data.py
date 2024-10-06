def combine_data(AIRS, FGS):
    FGS_column = FGS.sum(axis=2)
    dataset = np.concatenate([AIRS, FGS_column[:, :, np.newaxis, :]], axis=2)
    dataset = dataset.sum(axis=3)
    return dataset
