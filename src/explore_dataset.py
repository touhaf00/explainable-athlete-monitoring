import os
from scipy.io import loadmat

root = "../SportDB2"

for sport in os.listdir(root):
    sport_path = os.path.join(root, sport)
    if not os.path.isdir(sport_path):
        continue

    print("\nSPORT:", sport)

    for subject in os.listdir(sport_path)[:2]:
        subject_path = os.path.join(sport_path, subject)
        if not os.path.isdir(subject_path):
            continue

        print("  SUBJECT:", subject)

        for crd in os.listdir(subject_path):
            crd_path = os.path.join(subject_path, crd)
            data_path = os.path.join(crd_path, "Data.mat")
            trnote_path = os.path.join(crd_path, "TrNote.txt")
            dem_path = os.path.join(crd_path, "Dem.txt")

            if os.path.exists(data_path):
                data = loadmat(data_path)
                D = data["Data"]
                print("    ", crd, "fields:", D.dtype.names)

            if os.path.exists(trnote_path):
                with open(trnote_path, "r", encoding="latin-1") as f:
                    print("      TrNote:", f.read().replace("\n", " | "))

        break