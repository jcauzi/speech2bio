import librosa
import soundfile as sf
import tqdm as tqdm
import os


data_path = "data/egyptian_fruit_bats/audio/"
target_path = "data/egyptian_fruit_bats/audio_rs/"

#data_path = "data/cbi/wav/"
#target_path = "data/cbi/wav_rs/"

file_path = "7600.wav"
file_path2 = "8004.wav"
file_path3 = "12524.wav"

n_steps = -32

for file in tqdm.tqdm(os.listdir(data_path)):
    if file not in os.listdir(target_path) :

#for file in ["XC31278.wav", "XC31289.wav", "XC10722.wav"] :

        y, sr = librosa.load(data_path+file)  # Keep the original sample rate

        # Pitch-shift to a lower frequency range (e.g., down by 4 octaves)
        # Number of semitones to shift (negative for downpitching)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        y_resampled = librosa.resample(y_shifted, orig_sr=sr, target_sr=16000)

        # Save the pitch-shifted audio
        sf.write(target_path+file, y_resampled, 16000)  # Save at the original sample rate