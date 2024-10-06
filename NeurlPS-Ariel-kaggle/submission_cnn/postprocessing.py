def postprocessing(predictions, predictions_std, index):
    '''
    combine the results to dataframe that meet the requirement.
    '''
    res = pd.concat([
        pd.DataFrame(predictions.clip(0, None), index=index, columns=[f"wl_{i}" for i in range(1, 284)]),
        pd.DataFrame(predictions_std, index=index, columns=[f"sigma_{i}" for i in range(1, 284)])
        ], axis=1)
    return res