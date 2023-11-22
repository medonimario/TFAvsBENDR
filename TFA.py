import mne
import os
import numpy as np
import pandas as pd

def import_data(folder_path, pair_nr):
    # Importing data for person 1 and person 2
    sample_data_folder = mne.datasets.sample.data_path()
    pair010_p1_coupled_file = (
        folder_path + f"pair{pair_nr}_p1_coupled.fif"
    )
    p1_coupled_epochs = mne.read_epochs(pair010_p1_coupled_file, preload=True)

    pair010_p1_uncoupled_file = (
        folder_path + f"pair{pair_nr}_p1_uncoupled.fif"
    )
    p1_uncoupled_epochs = mne.read_epochs(pair010_p1_uncoupled_file, preload=True)

    pair010_p2_coupled_file = (
        folder_path + f"pair{pair_nr}_p2_coupled.fif"
    )
    p2_coupled_epochs = mne.read_epochs(pair010_p2_coupled_file, preload=True)

    pair010_p2_uncoupled_file = (
        folder_path + f"pair{pair_nr}_p2_uncoupled.fif"
    )
    p2_uncoupled_epochs = mne.read_epochs(pair010_p2_uncoupled_file, preload=True)
    return p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs

def compute_power(p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs, _19_channels=False):
        
    # Setup innitial parameters
    freqs = np.arange(1,31,1)  # frequencies from 1-30Hz
    n_cycles = freqs  # different number of cycle per frequency
    time_bandwidth = 2.0 # only 1 taper
    # Selecting the channels
    picks = None
    if _19_channels:
        picks = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

    # Compute power for each condition
    epochsTFR_p1_uncoupled = mne.time_frequency.tfr_multitaper(
        p1_uncoupled_epochs,
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=False
    )
    epochsTFR_p1_coupled = mne.time_frequency.tfr_multitaper(
        p1_coupled_epochs,
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=False
    )
    epochsTFR_p2_uncoupled= mne.time_frequency.tfr_multitaper(
        p2_uncoupled_epochs,
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=False
    )
    epochsTFR_p2_coupled = mne.time_frequency.tfr_multitaper(
        p2_coupled_epochs,
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        return_itc=False,
        average=False
    )
    print(epochsTFR_p1_coupled.data.shape)
    print(epochsTFR_p1_uncoupled.data.shape)
    print(epochsTFR_p2_coupled.data.shape)
    print(epochsTFR_p2_uncoupled.data.shape)

    return epochsTFR_p1_uncoupled, epochsTFR_p1_coupled, epochsTFR_p2_uncoupled, epochsTFR_p2_coupled

def average_over_timepoints(data, avg_over=256):
    # Calculate the new number of time points after averaging
    new_timepoints = data.shape[-1] // avg_over
    # Reshape data to divide the time axis into chunks of size `avg_over`
    reshaped_data = data.reshape(*data.shape[:-1], new_timepoints, avg_over)
    # Average over the last axis to reduce the number of time points
    avg_data = reshaped_data.mean(axis=-1)
    return avg_data

def epochsTFR_to_df(epochsTFR_p1_uncoupled, epochsTFR_p1_coupled, epochsTFR_p2_uncoupled, epochsTFR_p2_coupled, _19_channels=False):
    # Convert to ndarray
    epochsTFR_p1_uncoupled_data = epochsTFR_p1_uncoupled.data
    epochsTFR_p1_coupled_data = epochsTFR_p1_coupled.data
    epochsTFR_p2_uncoupled_data = epochsTFR_p2_uncoupled.data
    epochsTFR_p2_coupled_data = epochsTFR_p2_coupled.data

    # Reduce dimensionality by averaging power across time
    avg_epochsTFR_p1_uncoupled_data = average_over_timepoints(epochsTFR_p1_uncoupled_data)
    avg_epochsTFR_p1_coupled_data = average_over_timepoints(epochsTFR_p1_coupled_data)
    avg_epochsTFR_p2_uncoupled_data = average_over_timepoints(epochsTFR_p2_uncoupled_data)
    avg_epochsTFR_p2_coupled_data = average_over_timepoints(epochsTFR_p2_coupled_data)

    # Combining the data into tensors
    # 1. Get the number of epochs for both uncoupled and coupled data
    n_epochs_uncoupled = avg_epochsTFR_p1_uncoupled_data.shape[0]
    n_epochs_coupled = avg_epochsTFR_p1_coupled_data.shape[0]

    # Initialize tensors
    if _19_channels:
        combined_uncoupled = np.zeros((n_epochs_uncoupled, 2, 19, 30, 3))
        combined_coupled = np.zeros((n_epochs_coupled, 2, 19, 30, 3))
    else:
        combined_uncoupled = np.zeros((n_epochs_uncoupled, 2, 64, 30, 3))
        combined_coupled = np.zeros((n_epochs_coupled, 2, 64, 30, 3))

    # 2. Loop for uncoupled data
    for epoch in range(n_epochs_uncoupled):
        combined_uncoupled[epoch, 0, :, :, :] = avg_epochsTFR_p1_uncoupled_data[epoch]
        combined_uncoupled[epoch, 1, :, :, :] = avg_epochsTFR_p2_uncoupled_data[epoch]

    # Loop for coupled data
    for epoch in range(n_epochs_coupled):
        combined_coupled[epoch, 0, :, :, :] = avg_epochsTFR_p1_coupled_data[epoch]
        combined_coupled[epoch, 1, :, :, :] = avg_epochsTFR_p2_coupled_data[epoch]
    
    # Flatten the tensors into 1D feature vectors
    flattened_uncoupled = combined_uncoupled.reshape(n_epochs_uncoupled, -1)
    flattened_coupled = combined_coupled.reshape(n_epochs_coupled, -1)

    # Create a DataFrame for both variants
    df_uncoupled = pd.DataFrame(flattened_uncoupled)
    df_uncoupled['label'] = 1  # Label for uncoupled

    df_coupled = pd.DataFrame(flattened_coupled)
    df_coupled['label'] = 2  # Label for coupled

    # Concatenate the two dataframes
    df = pd.concat([df_uncoupled, df_coupled], axis=0).reset_index(drop=True)
    print(df.shape)
    return df

if __name__ == "__main__":
    folder_path = "C:/1_University/Thesis/Scripts/SC_epochs/"
    _19_channels = True
    #pair_nr = 10
    TF_df = []
    pairs = ["003", "004", "005", "007", "008", "009", "010", "011","012", "013", "014", "016", "017", "018", "019", "020", "022", "023", "024", "025", "027"]
    for pair_nr in pairs:
        # Import data
        p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs = import_data(folder_path, pair_nr)

        # Compute power
        epochsTFR_p1_uncoupled, epochsTFR_p1_coupled, epochsTFR_p2_uncoupled, epochsTFR_p2_coupled = compute_power(p1_uncoupled_epochs, p1_coupled_epochs, p2_uncoupled_epochs, p2_coupled_epochs, _19_channels)

        # Convert to dataframe
        df = epochsTFR_to_df(epochsTFR_p1_uncoupled, epochsTFR_p1_coupled, epochsTFR_p2_uncoupled, epochsTFR_p2_coupled, _19_channels)
        print("Pair " + pair_nr + "df shape: " + str(df.shape))
        # Save dataframe
        if _19_channels:
            df.to_csv(f"C:/1_University/Thesis/Scripts/TF_df_19_new/pair{pair_nr}_df.csv", index=False)
        else:
            df.to_csv(f"C:/1_University/Thesis/Scripts/TF_df/pair{pair_nr}_df.csv", index=False)