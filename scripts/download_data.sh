#! /bin/env bash

mkdir -p data
python scripts/watkins.py #done and validated
python scripts/bats.py #done and validated
python scripts/cbi.py #not done but unvalid (run validate and check md5 files + ask Hagiwara ?)
python scripts/humbugdb.py #done and validated
python scripts/dogs.py #done and validated
python scripts/dcase.py #done and validated
python scripts/enabirds.py #done and validated
mkdir data/hiceas
wget https://storage.googleapis.com/ml-bioacoustics-datasets/hiceas_1-20_minke-detection.zip -O data/hiceas/hiceas.zip #done and validated
unzip data/hiceas/hiceas.zip -d data/hiceas #done and validated
python scripts/rfcx.py #done and validated
python scripts/hainan_gibbons.py #done and validated
python scripts/esc50.py #done and validated
python scripts/speech_commands.py #done and validated
python scripts/validate_data.py #done
