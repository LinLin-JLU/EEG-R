
import os
import numpy as np
import pywt
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif


# =============================
# Configuration
# =============================
DATA_DIR = './data/'              # EEG data directory (user should provide)
SAVE_DIR = './results/'           # Output directory
WAVELET = 'cmor'                  # Complex Morlet wavelet
SCALES = np.arange(1, 128)
K = 0.7                         # Dynamic threshold parameter

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================
# Helper functions
# =============================
def load_eeg_data(file_path: str) -> np.ndarray:
    """
    Load EEG data from a .npy file.
    Users can modify this function to match their dataset structure.
    Expected shape: (samples, channels)
    """
    return np.load(file_path)


def compute_coherence_matrix(channel_data, wavelet, scales):
    """
    Compute the coherence matrix between all channel pairs using CWT.
    """
    n_channels = len(channel_data)
    C = np.zeros((n_channels, n_channels))

    for i in tqdm(range(n_channels), desc="Computing coherence"):
        for j in range(i + 1, n_channels):
            ch1, ch2 = channel_data[i], channel_data[j]

            coeffs1, _ = pywt.cwt(ch1, scales, wavelet)
            coeffs2, _ = pywt.cwt(ch2, scales, wavelet)

            Wxy = np.abs(np.mean(coeffs1 * np.conj(coeffs2), axis=1))
            mean_coherence = np.mean(Wxy)

            C[i, j] = mean_coherence
            C[j, i] = mean_coherence

    return C


def compute_dynamic_threshold(C, K):
    """
    Compute a dynamic threshold based on mean and standard deviation.
    """
    upper_tri = C[np.triu_indices_from(C, k=1)]
    c_mean = np.mean(upper_tri)
    c_std = np.std(upper_tri)
    return c_mean + K * c_std


def select_channels(mi, C, T):
    """
    Channel selection based on mutual information and coherence redundancy.
    """
    n_channels = len(mi)
    selected = set(range(n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if C[i, j] >= T:  # high redundancy
                if mi[i] >= mi[j]:
                    selected.discard(j)
                else:
                    selected.discard(i)
    return sorted(list(selected))


# =============================
# Main processing
# =============================
def main():
    for file_name in os.listdir(DATA_DIR):
        if not file_name.endswith('.npy'):
            continue

        file_path = os.path.join(DATA_DIR, file_name)
    

        # Step 1: Load EEG data
        X = load_eeg_data(file_path)  # shape = (samples, channels)
        n_samples, n_channels = X.shape
   

        # Step 2: Prepare per-channel sequences
        channel_data = [X[:, ch].reshape(-1) for ch in range(n_channels)]

        # Step 3: Compute coherence matrix
        C = compute_coherence_matrix(channel_data, WAVELET, SCALES)
        np.savetxt(os.path.join(SAVE_DIR, f"{file_name}_coherence_matrix.csv"), C, delimiter=",")

        # Step 4: Dynamic threshold
        T = compute_dynamic_threshold(C, K)
        print(f"Dynamic threshold T = {T:.4f}")

        # Step 5: Load binary labels (user must provide)
        # Replace these lines with your label loading logic
        y_valence = np.load('./labels/valence.npy')
        y_arousal = np.load('./labels/arousal.npy')

        # Step 6: Mutual information
        mi_valence = mutual_info_classif(X, y_valence, discrete_features=False)
        mi_arousal = mutual_info_classif(X, y_arousal, discrete_features=False)

        # Step 7: Channel selection
        selected_valence = select_channels(mi_valence, C, T)
        selected_arousal = select_channels(mi_arousal, C, T)

        print("Valence channels selected:", len(selected_valence))
        print("Arousal channels selected:", len(selected_arousal))

        # Step 8: Save selection results
        result_file = os.path.join(SAVE_DIR, f"{file_name}_selected_channels.txt")
        with open(result_file, "w") as f:
            f.write("Valence selected channels:\n")
            f.write(",".join(map(str, selected_valence)) + "\n\n")
            f.write("Arousal selected channels:\n")
            f.write(",".join(map(str, selected_arousal)))

        print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
