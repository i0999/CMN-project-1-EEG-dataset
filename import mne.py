import mne
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
#from mne.datasets import eegbci
#mne.datasets.eegbci.load_data(subjects=list(range(1, 110)), runs=list(3,4,7,8,11,12), path=None, force_update=False, update_path=None, base_url='https://physionet.org/files/eegmmidb/1.0.0/', verbose=None)

def load_left_hand_epochs(subject=1, tmin=0.0, tmax=4.0):
    """
    Returns:
        epochs_left_real: left hand REAL movement epochs
        epochs_left_imag: left hand IMAGINED movement epochs
    """

    # Real left vs right hand runs
    real_runs = [3, 7]

    # Imagined left vs right hand runs
    imag_runs = [4, 8]

    real_files = mne.datasets.eegbci.load_data(subjects=[subject], runs=real_runs)

    raw_real = concatenate_raws([
        read_raw_edf(f, preload=True, verbose=False) for f in real_files
    ])

    events_real, event_id_real = mne.events_from_annotations(raw_real)

    # Keep ONLY left hand trials: T1
    epochs_left_real = mne.Epochs(
        raw_real,
        events_real,
        event_id={"left_real": event_id_real["T1"]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    # -------------------------
    # Load IMAGINED runs
    # -------------------------
    imag_files = mne.datasets.eegbci.load_data(subjects=[subject], runs=imag_runs)

    raw_imag = concatenate_raws([
        read_raw_edf(f, preload=True, verbose=False) for f in imag_files
    ])

    events_imag, event_id_imag = mne.events_from_annotations(raw_imag)

    # Keep ONLY left hand trials: T1
    epochs_left_imag = mne.Epochs(
        raw_imag,
        events_imag,
        event_id={"left_imag": event_id_imag["T1"]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    return epochs_left_real, epochs_left_imag

epochs_left_real, epochs_left_imag = load_left_hand_epochs(subject=1)

epochs_real = epochs_left_real
epochs_imag = epochs_left_imag

X_real = epochs_real.get_data()
X_imag = epochs_imag.get_data()

# labels: 0 = real, 1 = imagined
y_real = [0] * len(X_real)
y_imag = [1] * len(X_imag)

X = np.concatenate([X_real, X_imag], axis=0)
y = np.array(y_real + y_imag)

print(X.shape, y.shape)


# Example:

print("REAL left-hand trials:", len(epochs_left_real))
print("IMAG left-hand trials:", len(epochs_left_imag))
print(epochs_left_real)
print(epochs_left_imag)
