import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = "dataset"
OUTPUT_PATH = "spectrograms"

# create output directory if it does not exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Starting audio preprocessing...\n")

# -----------------------------
# Loop through each bird folder
# -----------------------------
for bird_species in os.listdir(DATASET_PATH):

    bird_folder = os.path.join(DATASET_PATH, bird_species)

    # skip if not a directory
    if not os.path.isdir(bird_folder):
        continue

    print("Processing species:", bird_species)

    # create species output folder
    output_species_folder = os.path.join(OUTPUT_PATH, bird_species)
    os.makedirs(output_species_folder, exist_ok=True)

    # -----------------------------
    # Process each audio file
    # -----------------------------
    for audio_file in os.listdir(bird_folder):

        if not (audio_file.endswith(".ogg") or audio_file.endswith(".wav")):
            continue

        audio_path = os.path.join(bird_folder, audio_file)

        try:

            # load audio file
            y, sr = librosa.load(audio_path)

            # generate log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                fmax=8000
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # create plot
            plt.figure(figsize=(3,3))
            librosa.display.specshow(
                mel_spec_db,
                sr=sr,
                x_axis="time",
                y_axis="mel"
            )

            plt.axis("off")

            # output file name
            output_file = audio_file.replace(".ogg", ".png").replace(".wav", ".png")
            save_path = os.path.join(output_species_folder, output_file)

            # save image
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

        except Exception as e:
            print("Error processing:", audio_path)
            print(e)

print("\nPreprocessing completed!")
print("Spectrograms saved in:", OUTPUT_PATH)