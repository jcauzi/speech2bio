from collections import defaultdict
import hashlib
import math
from pathlib import Path

from plumbum import local
import torch
import torchaudio
import csv


def get_wav_length_in_secs(path):
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def get_md5(file_name):
    with open(file_name, mode='rb') as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)

        return file_hash.hexdigest()


# def check_md5(file_name, md5):
#     if md5 != get_md5(file_name):
#         assert False, f'md5 for {file_name} does not match'

#check_md5 with file_name saver 
def check_md5(file_name, md5, csv_file='md5_failures.csv'):
    # Compare the provided MD5 with the file's actual MD5
    if md5 != get_md5(file_name):
        # Open the CSV file in append mode to add the failed file names
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the failed file name to the CSV
            writer.writerow([file_name])
        print(f'{file_name} has been saved in {csv_file} due to MD5 mismatch.')


def divide_waveform_to_chunks(path, target_dir, chunk_size, target_sample_rate):
    waveform, sample_rate = torchaudio.load(path)
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    if sample_rate != target_sample_rate:
        transform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = transform(waveform)
    num_samples = waveform.shape[1]
    num_seconds = num_samples / target_sample_rate
    num_chunks = math.ceil(num_seconds / chunk_size)
    target_paths = []
    for chunk in range(num_chunks):
        target_path = Path(target_dir) / f'{Path(path).stem}.{chunk:03d}.wav'
        st_sample = int(chunk * chunk_size * target_sample_rate)
        ed_sample = int((chunk + 1) * chunk_size * target_sample_rate)
        torchaudio.save(
            target_path,
            waveform[:, st_sample:ed_sample],
            sample_rate=target_sample_rate
        )
        target_paths.append(str(target_path))

    return target_paths


def divide_annotation_to_chunks(annotations, chunk_size):
    chunks = defaultdict(list)
    for anon in annotations:
        st, ed = anon['st'], anon['ed']     # in seconds
        st_chunk, ed_chunk = int(st // chunk_size), int(ed // chunk_size)

        for chunk in range(st_chunk, ed_chunk + 1):
            if chunk == st_chunk and chunk == ed_chunk:
                local_st, local_ed = st - chunk * chunk_size, ed - chunk * chunk_size
            elif chunk == st_chunk:
                local_st, local_ed = st - chunk * chunk_size, chunk_size
            elif chunk == ed_chunk:
                local_st, local_ed = 0, ed - chunk * chunk_size
            else:
                local_st, local_ed = 0, chunk_size

            new_anon = dict(anon)
            new_anon['st'], new_anon['ed'] = local_st, local_ed
            chunks[chunk].append(new_anon)

    return chunks
