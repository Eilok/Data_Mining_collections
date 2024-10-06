def preprocess_data(index, CHUNKS_SIZE, path_folder, train_adc_info, axis_info):
    '''
    conduct the data preprocessing
    '''
    DO_READ = True # readout correction
    DO_MASK = True  # mask dead and hot pixels
    DO_THE_NL_CORR = False  # linear correction
    DO_DARK = True # dark current correction
    DO_FLAT = True  # flat correction
    TIME_BINNING = True  # time binning

    cut_inf, cut_sup = 39, 321  # cut along the wavelength axis between pixels 39 to 321
    l = cut_sup - cut_inf

    # create the whole dataset
    data_train_AIRS = np.zeros((len(index), 187, 282, 32))
    data_train_FGS = np.zeros((len(index), 187, 32, 32))

    # deal with each data chunk
    for n, index_chunk in enumerate(tqdm(index)):
        AIRS_CH0_clean = np.zeros((CHUNKS_SIZE, 11250, 32, l))
        FGS1_clean = np.zeros((CHUNKS_SIZE, 135000, 32, 32))
        
        # load signal and conduct ADC, readout, dark, flat, dead, linear correction
        
        i = 0
        # deal with AIRS-CH0
        df = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_signal.parquet'))
        signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 356))
        gain = train_adc_info['AIRS-CH0_adc_gain'].loc[index_chunk]
        offset = train_adc_info['AIRS-CH0_adc_offset'].loc[index_chunk]
        signal = ADC_convert(signal, gain, offset)
        # dt_airs load signal integration time information
        dt_airs = axis_info['AIRS-CH0-integration_time'].dropna().values
        dt_airs[1::2] += 0.1
        chopped_signal = signal[:, :, cut_inf:cut_sup]
        del signal, df
            
        # clean AIRS data
        readout = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_calibration/read.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dark = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_calibration/dark.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        dead_airs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_calibration/dead.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        # linear_corr = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_calibration/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        
        if DO_READ:
            chopped_signal = readout_correct(chopped_signal, readout)
            AIRS_CH0_clean[i] = chopped_signal
        else:
            AIRS_CH0_clean[i] = chopped_signal
        del readout

        if DO_MASK:
            chopped_signal = mask_hot_dead(chopped_signal, dead_airs, dark)
            AIRS_CH0_clean[i] = chopped_signal
        else:
            AIRS_CH0_clean[i] = chopped_signal
            
        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr, AIRS_CH0_clean[i])
            AIRS_CH0_clean[i,:, :, :] = linear_corr_signal
        # del linear_corr
        
        if DO_DARK: 
            cleaned_signal = clean_dark(AIRS_CH0_clean[i], dead_airs, dark, dt_airs)
            AIRS_CH0_clean[i] = cleaned_signal
        else: 
            pass
        del dark
        
        # deal with FGS1 signal
        df = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_signal.parquet'))
        fgs_signal = df.values.astype(np.float64).reshape((df.shape[0], 32, 32))
        
        FGS1_gain = train_adc_info['FGS1_adc_gain'].loc[index_chunk]
        FGS1_offset = train_adc_info['FGS1_adc_offset'].loc[index_chunk]
        
        fgs_signal = ADC_convert(fgs_signal, FGS1_gain, FGS1_offset)
        dt_fgs1 = np.ones(len(fgs_signal))*0.1
        dt_fgs1[1::2] += 0.1
        chopped_FGS1 = fgs_signal
        del fgs_signal, df
        
        # clean FGS1 data
        readout = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_calibration/read.parquet')).values.astype(np.float64).reshape((32, 32))
        dark = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_calibration/dark.parquet')).values.astype(np.float64).reshape((32, 32))
        dead_fgs1 = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_calibration/dead.parquet')).values.astype(np.float64).reshape((32, 32))
        # linear_corr = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_calibration/linear_corr.parquet')).values.astype(np.float64).reshape((6, 32, 32))
        
        if DO_READ:
            chopped_FGS1 = readout_correct(chopped_FGS1, readout)
            FGS1_clean[i] = chopped_FGS1
        else:
            FGS1_clean[i] = chopped_FGS1
        del readout
        
        if DO_MASK:
            chopped_FGS1 = mask_hot_dead(chopped_FGS1, dead_fgs1, dark)
            FGS1_clean[i] = chopped_FGS1
        else:
            FGS1_clean[i] = chopped_FGS1

        if DO_THE_NL_CORR: 
            linear_corr_signal = apply_linear_corr(linear_corr, FGS1_clean[i])
            FGS1_clean[i,:, :, :] = linear_corr_signal
        # del linear_corr
        
        if DO_DARK: 
            cleaned_signal = clean_dark(FGS1_clean[i], dead_fgs1, dark, dt_fgs1)
            FGS1_clean[i] = cleaned_signal
        else: 
            pass
        del dark
            
        # CDS for whole chunk
        AIRS_cds = get_cds(AIRS_CH0_clean)
        FGS1_cds = get_cds(FGS1_clean)
        
        del AIRS_CH0_clean, FGS1_clean
        
        # flat field correction
        for i in range (CHUNKS_SIZE):
            flat_airs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/AIRS-CH0_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
            flat_fgs = pd.read_parquet(os.path.join(path_folder,f'test/{index_chunk}/FGS1_calibration/flat.parquet')).values.astype(np.float64).reshape((32, 32))
            if DO_FLAT:
                corrected_AIRS_cds = correct_flat_field(flat_airs, dead_airs, AIRS_cds[i])
                AIRS_cds[i] = corrected_AIRS_cds
                corrected_FGS1_cds = correct_flat_field(flat_fgs, dead_fgs1, FGS1_cds[i])
                FGS1_cds[i] = corrected_FGS1_cds
            else:
                pass

        # Time Binning to reduce space
        if TIME_BINNING:
            AIRS_cds_binned = bin_obs(AIRS_cds, binning=30)
            FGS1_cds_binned = bin_obs(FGS1_cds, binning=30*12)
        else:
            AIRS_cds = AIRS_cds.transpose(0,1,3,2) ## this is important to make it consistent for flat fielding, but you can always change it
            AIRS_cds_binned = AIRS_cds
            FGS1_cds = FGS1_cds.transpose(0,1,3,2)
            FGS1_cds_binned = FGS1_cds
        
        del AIRS_cds, FGS1_cds
        
        # save data
        data_train_AIRS[n] = AIRS_cds_binned
        data_train_FGS[n] = FGS1_cds_binned
        
    return data_train_AIRS, data_train_FGS