import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from spafe.features.lfcc import lfcc as get_lfcc

def extract_lfcc(file_path, max_frames=400):
    audio, sr = librosa.load(file_path, sr=16000)
    features = get_lfcc(sig=audio, fs=sr, num_ceps=20, nfilts=40)
    
    # audio pre emphasized
    audio = librosa.effects.preemphasis(audio, coef=0.97)

    features = features.T  # transpose to (coeffiecients, frames)

    cols = features.shape[1]
    if cols < max_frames:
        # pad with zeros if clip too short
        padding = max_frames - cols
        features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
    else:
        # truncate if clip too long
        features = features[:, :max_frames]

    return features

class ASVDataset(Dataset):
    def __init__(self, protocol_map, base_path, max_frames=400):
        """
        Args:
            protocol_map (dict): Your {filename: label} dictionary
            base_path (str): Path to your /flac/ folder on the T7
            max_frames (int): Our fixed width of 400
        """
        self.protocol_map = protocol_map
        self.filenames = list(protocol_map.keys())
        self.base_path = base_path
        self.max_frames = max_frames

    def __len__(self):
        # tells pytorch how many files
        return len(self.filenames)

    def __getitem__(self, idx):
        # get filename and label
        file_name = self.filenames[idx]
        label = self.protocol_map[file_name]
        
        # build full path
        full_path = os.path.join(self.base_path, f"{file_name}.flac")
        
        # get (20,400) matrix from extract lfcc
        features = extract_lfcc(full_path, max_frames=self.max_frames)
        
        # convert to pytorch tensor and add channel dimension
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        label_tensor = torch.tensor(label).long()
        
        return features_tensor, label_tensor
    
# this maps the protocol files to a dictionary of filename: label (1 for bonafide, 0 for spoof)
# this will help the model easily access the labels to understand which files are bonafide and which are spoof during training.
def map_labels(protocol_path, label_map):
    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()

            filename = parts[1]
            label = parts[4]

            label_bin = 1 if label == "bonafide" else 0
            label_map[filename] = label_bin

    print(f"Total files mapped: {len(label_map)}")

def create_npy_bundle(dataset, out_prefix="train_data"):
    """
    saves dataset as two numpy files
    """
    num_samples = len(dataset)
    # pre allocate memory
    all_features = np.zeros((num_samples, 1, 20, 400), dtype=np.float32)
    all_labels = np.zeros((num_samples,), dtype=np.int64)

    print(f"Starting bundle creation for {num_samples} samples...")

    for i in range(num_samples):
        try:
            feat, label = dataset[i]
            all_features[i] = feat.numpy()
            all_labels[i] = label.item()
            
            if i % 500 == 0:
                print(f"Progress: {i}/{num_samples} ({(i/num_samples)*100:.1f}%)")
        except Exception as e:
            print(f"Error at index {i}: {e}")

    print("Saving files to disk...")
    np.save(f"{out_prefix}_x.npy", all_features)
    np.save(f"{out_prefix}_y.npy", all_labels)
    print(f"Done! Created {out_prefix}_x.npy and {out_prefix}_y.npy")



protocol_path = '/Volumes/T7/ASVspoof dataset/archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
label_map = {}
map_labels(protocol_path=protocol_path, label_map=label_map)

train_audio_path = '/Volumes/T7/ASVspoof dataset/archive/LA/LA/ASVspoof2019_LA_eval/flac'

# initialize the pipeline
train_dataset = ASVDataset(
    protocol_map=label_map, # The dict we built earlier
    base_path=train_audio_path,
    max_frames=400
)

create_npy_bundle(train_dataset, out_prefix="ASV_evl_LA")