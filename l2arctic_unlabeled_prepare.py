import sys
import os
import json
from speechbrain.dataio.dataio import read_audio
from glob import glob
import re
import copy
from collections import defaultdict


SAMPLERATE = 44100
phn_set="data/arpa_phonemes"
def process_arpa_phoneme(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    arpa_phonemes= []
    for line in lines:
        items = line.strip().split()
        arpa_phonemes.append(items[0])
    return arpa_phonemes

ARPA_PHONEMES = process_arpa_phoneme(phn_set)

def prepare_l2arctic_unlabeled(
    data_folder,
    save_json_train="train_l2arctic_unlabeled.json",
    labeled_json="train_l2arctic.json",
    metadata_l2arctic="data/metadata_l2arctic",
    test_spks=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK',],
):
    if not os.path.exists(os.path.dirname(save_json_train)):
        os.makedirs(os.path.dirname(save_json_train))
    total_spks = []
    with open(metadata_l2arctic, 'r') as reader:
        for ii, line in enumerate(reader):
            if ii == 0:
                # Skip the header
                continue
            name, dialect, gender = line.split()
            total_spks.append(name)
    train_spks = [x for x in total_spks if x not in test_spks]

    with open(labeled_json, "r") as json_file:
        labeled_data = json.load(json_file)

    spks_to_solve = [
        (save_json_train, train_spks),
    ]
    for split, spks in spks_to_solve:
        make_json(data_folder, split, spks, labeled_data)

def make_json(data_folder, split, spks, labeled_data):
    """
    check whether the wav is presented in labled_data
    we only keep those wav that were not labled.
    """
    print("Creating {}".format(split))

    json_data = defaultdict(dict)
    for spk in spks:
        spk_data = get_data_from_spk(data_folder, spk, labeled_data)
        json_data.update(spk_data)
    with open(split, mode="w") as json_f:
        json.dump(json_data, json_f, indent=2)

    print(f"{split} successfully created!")

def get_data_from_spk(data_folder, spk, labeled_data):
    wav_dir = os.path.join(data_folder, spk, 'wav')
    tg_dir = os.path.join(data_folder, spk, 'annotation')
    text_dir = os.path.join(data_folder, spk, 'transcript')
    spk_data = defaultdict(dict)
    for wav_file in glob(os.path.join(wav_dir, "*.wav")):
        if wav_file in labeled_data:
            continue

        basename = os.path.basename(wav_file).split(".")[0]
        text_file = os.path.join(text_dir, basename + '.txt')
        utt_data = get_data_from_utt(wav_file, text_file, spk)
        spk_data.update(utt_data)
    return spk_data

def get_data_from_utt( wav_file, text_file, spk):
    utt_data = {}
    utt_data[wav_file] = {}
    utt_data[wav_file]["wav"] = wav_file
    # Reading the signal (to retrieve duration in seconds)
    signal = read_audio(wav_file)
    duration = len(signal) / SAMPLERATE
    utt_data[wav_file]["duration"] = duration
    utt_data[wav_file]["spk_id"] = spk

    with open(text_file, "r") as reader:
        text = reader.readline()
    utt_data[wav_file]["wrd"] = text
    return utt_data



if __name__ == "__main__":

    prepare_l2arctic_unlabeled(
        data_folder=sys.argv[1],
    save_json_train="data/train_unlabeled.json",
    labeled_json="data/train.json",
    metadata_l2arctic="data/metadata_l2arctic",
    test_spks=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK',],
)


