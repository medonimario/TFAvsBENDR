import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoreject import AutoReject
ar = AutoReject()

def import_data(folder_path, pair_nr):
    # Importing data for person 1 and person 2
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file_p1 = (
        folder_path+f"pair{pair_nr}_p1_pre-processed.fif"
    )
    p1_epochs = mne.read_epochs(sample_data_raw_file_p1, preload=False)

    sample_data_raw_file_p2 = (
        folder_path+f"pair{pair_nr}_p2_pre-processed.fif"
    )
    p2_epochs = mne.read_epochs(sample_data_raw_file_p2, preload=False)
    return p1_epochs, p2_epochs

def drop_second_epoch(p1_epochs, p2_epochs):
    p1_epochs = p1_epochs[::2]
    p2_epochs = p2_epochs[::2]
    return p1_epochs, p2_epochs

def divide_epochs(p1_epochs, p2_epochs):
    '''Divides the epochs into sub-epochs of 3 seconds each, with 2 seconds omitted from the start of each epoch.'''

    p1_epochs_uncoupled = p1_epochs["uncoupled"].copy()
    p1_epochs_coupled = p1_epochs["coupled"].copy()
    p2_epochs_uncoupled = p2_epochs["uncoupled"].copy()
    p2_epochs_coupled = p2_epochs["coupled"].copy()

    print("Innitial ammount of epochs:")
    print("P1 Uncoupled: "+str(len(p1_epochs_uncoupled)))
    print("P1 Coupled: "+str(len(p1_epochs_coupled)))
    print("P2 Uncoupled: "+str(len(p2_epochs_uncoupled)))
    print("P2 Coupled: "+str(len(p2_epochs_coupled)))

    # Define the desired duration for sub-epochs
    sub_epoch_duration = 3.0  # in seconds

    # Define the duration to omit from the start of each epoch
    omit_duration = 2.0  # in seconds


    # Pick only EEG channels from the existing info object
    eeg_picks = mne.pick_types(p1_epochs_uncoupled.info, meg=False, eeg=True)

    epochs = [p1_epochs_uncoupled, p1_epochs_coupled, p2_epochs_uncoupled, p2_epochs_coupled]
    sub_epochs = []

    for epoch in epochs:
        # Initialize an empty list to store the sub-epochs
        sub_epochs_list = []

        # Iterate over each original epoch
        for epoch_idx in range(len(epoch)):
            # Get the data for the current epoch
            data = epoch[epoch_idx].get_data()

            #Calculate the number of samples to omit
            omit_samples = int(omit_duration * epoch.info['sfreq'])
            
            # Calculate the number of sub-epochs within the remaining duration
            remaining_samples = data.shape[2] - omit_samples
            num_sub_epochs = remaining_samples // int(sub_epoch_duration * epoch.info['sfreq'])

            # Iterate to create sub-epochs
            for sub_epoch_idx in range(num_sub_epochs):
                # Calculate the start and end sample indices for the sub-epoch
                start_sample = omit_samples + sub_epoch_idx * int(sub_epoch_duration * epoch.info['sfreq'])
                end_sample = omit_samples + (sub_epoch_idx + 1) * int(sub_epoch_duration * epoch.info['sfreq'])

                # Create a new info object for EEG data
                sub_info = mne.pick_info(epoch.info, eeg_picks)

                # Create a new Epochs object for the sub-epoch
                sub_epoch_data = data[:, :,start_sample:end_sample]
                sub_epoch = mne.EpochsArray(sub_epoch_data, sub_info)

                # Append the sub-epoch to the list
                sub_epochs_list.append(sub_epoch)
            
        # Create a new 'sub_epochs' Epochs object from the list of sub-epochs
        sub_epochs.append(mne.concatenate_epochs(sub_epochs_list))

    p1_subepochs_uncoupled = sub_epochs[0]
    p1_subepochs_coupled = sub_epochs[1]
    p2_subepochs_uncoupled = sub_epochs[2]
    p2_subepochs_coupled = sub_epochs[3]

    print("Segmented ammount of epochs:")
    print("P1 Uncoupled: "+str(len(p1_subepochs_uncoupled)))
    print("P1 Coupled: "+str(len(p1_subepochs_coupled)))
    print("P2 Uncoupled: "+str(len(p2_subepochs_uncoupled)))
    print("P2 Coupled: "+str(len(p2_subepochs_coupled)))

    return p1_subepochs_uncoupled, p1_subepochs_coupled, p2_subepochs_uncoupled, p2_subepochs_coupled      

def remove_bad_trials(p1_subepochs_uncoupled, p1_subepochs_coupled, p2_subepochs_uncoupled, p2_subepochs_coupled, plot_folder_path, pair_nr):
    '''Removes bad trials from the sub-epochs using autoreject.'''

    p1_subepochs_uncoupled.load_data()
    p1_subepochs_coupled.load_data()
    p2_subepochs_uncoupled.load_data()
    p2_subepochs_coupled.load_data()

    #Use autoreject to get reject logs fro the subepochs that should be ommited
    p1_subepochs_uncoupled_clean, p1_uncoupled_reject_log = ar.fit_transform(p1_subepochs_uncoupled, return_log=True)  
    p1_subepochs_coupled_clean, p1_coupled_reject_log = ar.fit_transform(p1_subepochs_coupled, return_log=True)
    p2_subepochs_uncoupled_clean, p2_uncoupled_reject_log = ar.fit_transform(p2_subepochs_uncoupled, return_log=True)
    p2_subepochs_coupled_clean, p2_coupled_reject_log = ar.fit_transform(p2_subepochs_coupled, return_log=True)

    #Plot reject logs
    plot_reject_log(p1_uncoupled_reject_log, p1_coupled_reject_log, p2_uncoupled_reject_log, p2_coupled_reject_log, plot_folder_path, pair_nr)

    #Plot bad epochs
    plot_bad_epochs(p1_uncoupled_reject_log, p1_coupled_reject_log, p2_uncoupled_reject_log, p2_coupled_reject_log, plot_folder_path, pair_nr)

    #Get the bad epochs from the reject logs
    uncoupled_bad_epochs = np.logical_or(p1_uncoupled_reject_log.bad_epochs, p2_uncoupled_reject_log.bad_epochs)
    coupled_bad_epochs = np.logical_or(p1_coupled_reject_log.bad_epochs, p2_coupled_reject_log.bad_epochs)

    #Get the good epochs from the bad epochs
    uncoupled_good_epochs = np.invert(uncoupled_bad_epochs)
    coupled_good_epochs = np.invert(coupled_bad_epochs)

    #Get the final epochs
    p1_subepochs_uncoupled_final = p1_subepochs_uncoupled[uncoupled_good_epochs]
    p1_subepochs_coupled_final = p1_subepochs_coupled[coupled_good_epochs]

    p2_subepochs_uncoupled_final = p2_subepochs_uncoupled[uncoupled_good_epochs]
    p2_subepochs_coupled_final = p2_subepochs_coupled[coupled_good_epochs]

    print("Final ammount of epochs:")
    print("P1 Uncoupled: "+str(len(p1_subepochs_uncoupled_final)))
    print("P1 Coupled: "+str(len(p1_subepochs_coupled_final)))
    print("P2 Uncoupled: "+str(len(p2_subepochs_uncoupled_final)))
    print("P2 Coupled: "+str(len(p2_subepochs_coupled_final)))

    return p1_subepochs_uncoupled_final, p1_subepochs_coupled_final, p2_subepochs_uncoupled_final, p2_subepochs_coupled_final

def safe_plot_reject_log(reject_log, orientation, plot_path):
    if reject_log is None:
        print("Reject log is None. Skipping plot.")
        return None
    if len(reject_log.bad_epochs) == 0:
        print("Reject log is empty. Skipping plot.")
        return None

    plot_object = reject_log.plot(orientation)
    if plot_object is not None:
        plot_object.savefig(plot_path)
        plt.close()  # Close the plot
    return plot_object

def plot_reject_log(p1_uncoupled_reject_log, p1_coupled_reject_log, p2_uncoupled_reject_log, p2_coupled_reject_log, plot_folder_path, pair_nr):
    safe_plot_reject_log(p1_uncoupled_reject_log, 'horizontal', plot_folder_path + f"{pair_nr}_p1_uncoupled_reject_log.png")
    safe_plot_reject_log(p1_coupled_reject_log, 'horizontal', plot_folder_path + f"{pair_nr}_p1_coupled_reject_log.png")
    safe_plot_reject_log(p2_uncoupled_reject_log, 'horizontal', plot_folder_path + f"{pair_nr}_p2_uncoupled_reject_log.png")
    safe_plot_reject_log(p2_coupled_reject_log, 'horizontal', plot_folder_path + f"{pair_nr}_p2_coupled_reject_log.png")


def safe_plot(epochs, reject_log, n_epochs=16, n_channels=64):
    if len(epochs) == 0:
        print("Epochs object is empty. Skipping plot.")
        return None
    if np.all(np.array(reject_log.bad_epochs) == False):
        print("All bad_epochs values are False. Skipping plot.")
        return None
    if max(reject_log.bad_epochs) >= len(epochs):
        print("Invalid bad epoch indices. Skipping plot.")
        return None
    
    plot_object = epochs[reject_log.bad_epochs].plot(n_epochs=n_epochs, n_channels=n_channels)
    plt.close()  # Close the plot
    return plot_object

def plot_bad_epochs(p1_uncoupled_reject_log, p1_coupled_reject_log, p2_uncoupled_reject_log, p2_coupled_reject_log, plot_folder_path, pair_nr):
    plot5 = safe_plot(p1_subepochs_uncoupled, p1_uncoupled_reject_log)
    plot6 = safe_plot(p1_subepochs_coupled, p1_coupled_reject_log)
    plot7 = safe_plot(p2_subepochs_uncoupled, p2_uncoupled_reject_log)
    plot8 = safe_plot(p2_subepochs_coupled, p2_coupled_reject_log)

    if plot5 is not None:
        plot5.savefig(plot_folder_path+f"rejected_epochs/{pair_nr}_p1_uncoupled_bad_epochs.png")
    if plot6 is not None:
        plot6.savefig(plot_folder_path+f"rejected_epochs/{pair_nr}_p1_coupled_bad_epochs.png")
    if plot7 is not None:
        plot7.savefig(plot_folder_path+f"rejected_epochs/{pair_nr}_p2_uncoupled_bad_epochs.png")
    if plot8 is not None:
        plot8.savefig(plot_folder_path+f"rejected_epochs/{pair_nr}_p2_coupled_bad_epochs.png")




def export_data(p1_subepochs_uncoupled_final, p1_subepochs_coupled_final, p2_subepochs_uncoupled_final, p2_subepochs_coupled_final, export_folder_path, pair_nr):
    p1_uncoupled_fname = export_folder_path + f"pair{pair_nr}_p1_uncoupled.fif"
    p1_coupled_fname = export_folder_path + f"pair{pair_nr}_p1_coupled.fif"
    p2_uncoupled_fname = export_folder_path + f"pair{pair_nr}_p2_uncoupled.fif"
    p2_coupled_fname = export_folder_path + f"pair{pair_nr}_p2_coupled.fif"

    p1_subepochs_uncoupled_final.save(fname=p1_uncoupled_fname, overwrite=True)
    p1_subepochs_coupled_final.save(fname=p1_coupled_fname, overwrite=True)
    p2_subepochs_uncoupled_final.save(fname=p2_uncoupled_fname, overwrite=True)
    p2_subepochs_coupled_final.save(fname=p2_coupled_fname, overwrite=True)

if __name__ == "__main__":
    pairs = ["003", "004", "005", "007", "008", "009", "010", "011","012", "013", "014", "016", "017", "018", "019", "020", "022", "023", "024", "025", "027"]
    for pair_nr in pairs:
        folder_path = "C:/1_University/Thesis/Scripts/Pre-processed data/"
        export_folder_path = "C:/1_University/Thesis/Scripts/SC_epochs/"
        plot_folder_path = "C:/1_University/Thesis/Scripts/SC_plots/"
        p1_epochs, p2_epochs = import_data(folder_path, pair_nr)
        p1_epochs, p2_epochs = drop_second_epoch(p1_epochs, p2_epochs)
        p1_subepochs_uncoupled, p1_subepochs_coupled, p2_subepochs_uncoupled, p2_subepochs_coupled = divide_epochs(p1_epochs, p2_epochs)
        p1_subepochs_uncoupled_final, p1_subepochs_coupled_final, p2_subepochs_uncoupled_final, p2_subepochs_coupled_final = remove_bad_trials(p1_subepochs_uncoupled, p1_subepochs_coupled, p2_subepochs_uncoupled, p2_subepochs_coupled, plot_folder_path, pair_nr)
        export_data(p1_subepochs_uncoupled_final, p1_subepochs_coupled_final, p2_subepochs_uncoupled_final, p2_subepochs_coupled_final, export_folder_path, pair_nr)

        