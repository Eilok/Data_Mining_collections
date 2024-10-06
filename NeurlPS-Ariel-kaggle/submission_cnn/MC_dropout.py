def unstandardizing(data, min_train_valid, max_train_valid):
    return data * (max_train_valid - min_train_valid) + min_train_valid

# conduct Monte Carlo Dropout prediction
def MC_dropout_WC(model, data, nb_dropout):
    data = torch.tensor(data, dtype=torch.float32)
    model.train() # set train mode
    predictions = torch.zeros(nb_dropout, data.size(0)) # store prediction results

    with torch.no_grad():
        for i in range(nb_dropout):
            output = model(data.unsqueeze(1))  # predict
            predictions[i, :] = output.flatten()

    return predictions

# Compute the uncertainties
def NN_uncertainty(model, x_test, targets_abs_max, T=5):
    model.eval()  # Set the model to evaluation mode
    predictions = []  # Save the predictions
    with torch.no_grad():
        for _ in range(T):
            # Move input to the appropriate device (CPU or GPU)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
            pred_norm = model(x_test_tensor).numpy()  # Make predictions
            pred = targets_norm_back(pred_norm, targets_abs_max)  # Reverse normalization
            predictions.append(pred)  # Store each prediction
    mean, std = np.mean(np.array(predictions), axis=0), np.std(np.array(predictions), axis=0)
    return mean, std 