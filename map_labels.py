import os

protocol_path = "/Volumes/T7/ASVspoof dataset/archive/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
label_map = {}

with open(protocol_path, "r") as f:
    for line in f:
        line = line.strip()
        parts = line.split()

        filename = parts[1]
        label = parts[4]

        label_bin = 1 if label == "bonafide" else 0
        label_map[filename] = label_bin