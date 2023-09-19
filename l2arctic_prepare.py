import sys
import os
import json
from speechbrain.dataio.dataio import read_audio
from glob import glob
from textgrid import TextGrid, IntervalTier
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

def prepare_l2arctic(
    data_folder,
    save_json_train="train_l2arctic.json",
    save_json_test="test_l2arctic.json",
    metadata_l2arctic="data/metadata_l2arctic",
    test_spks=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK',]
):
    if not os.path.exists(os.path.dirname(save_json_train)):
        os.makedirs(os.path.dirname(save_json_train))
    if not os.path.exists(os.path.dirname(save_json_test)):
        os.makedirs(os.path.dirname(save_json_test))
    total_spks = []
    with open(metadata_l2arctic, 'r') as reader:
        for ii, line in enumerate(reader):
            if ii == 0:
                # Skip the header
                continue
            name, dialect, gender = line.split()
            total_spks.append(name)
    train_spks = [x for x in total_spks if x not in test_spks]

    spks_to_solve = [
        (save_json_train, train_spks),
        (save_json_test, test_spks)
    ]
    for split, spks in spks_to_solve:
        make_json(data_folder, split, spks)

def make_json(data_folder, split, spks):
    print("Creating {}".format(split))

    json_data = defaultdict(dict)
    for spk in spks:
        spk_data = get_data_from_spk(data_folder, spk)
        json_data.update(spk_data)
    with open(split, mode="w") as json_f:
        json.dump(json_data, json_f, indent=2)

    print(f"{split} successfully created!")

def get_data_from_spk(data_folder, spk):
    wav_dir = os.path.join(data_folder, spk, 'wav')
    tg_dir = os.path.join(data_folder, spk, 'annotation')
    text_dir = os.path.join(data_folder, spk, 'transcript')

    spk_data = defaultdict(dict)
    for tg_file in glob(os.path.join(tg_dir, "*.TextGrid")):

        tg = TextGrid()
        try:
            tg.read(tg_file)
        except ValueError:
            continue

        basename = os.path.basename(tg_file).split(".")[0]
        wav_file = os.path.join(wav_dir, basename + ".wav")
        text_file = os.path.join(text_dir, basename + '.txt')
        utt_data = get_data_from_utt(tg, wav_file, text_file, spk)
        spk_data.update(utt_data)
    return spk_data

def get_data_from_utt(tg, wav_file, text_file, spk):
    utt_data = {}
    utt_data[wav_file] = {}
    utt_data[wav_file]["wav"] = wav_file
    # Reading the signal (to retrieve duration in seconds)
    signal = read_audio(wav_file)
    duration = len(signal) / SAMPLERATE
    utt_data[wav_file]["duration"] = duration
    utt_data[wav_file]["spk_id"] = spk
    ## To keep original human annotation, set `keep_artifical_sil=True`, `rm_repetitive_sil=False`
    ## this preserve the original alignment within the annotations
    cano_phns_align, perc_phns_align = get_phonemes(tg, keep_artificial_sil=True, rm_repetitive_sil=False)
    utt_data[wav_file]["canonical_aligned"] = cano_phns_align
    utt_data[wav_file]["perceived_aligned"] = perc_phns_align
    ## To get training target phones, set `keep_artifical_sil=False`, `rm_repetitive_sil=True`
    ## this apply some preprocessing on the perceived phones, i.e. rm artifical and repetitive sil
    _, target_phns = get_phonemes(tg, keep_artificial_sil=False, rm_repetitive_sil=True)
    utt_data[wav_file]["perceived_train_target"] = target_phns

    with open(text_file, "r") as reader:
        text = reader.readline()
    utt_data[wav_file]["wrd"] = text
    return utt_data

def get_phonemes(tg, keep_artificial_sil=False, rm_repetitive_sil=True):
    phone_tier = tg.getFirst("phones")
    perceived_phones = normalize_tier_mark(phone_tier, "NormalizePhonePerceived", keep_artificial_sil)
    canonical_phones = normalize_tier_mark(phone_tier, "NormalizePhoneCanonical", keep_artificial_sil)
    canonical_phones = tier_to_list(canonical_phones)
    perceived_phones = tier_to_list(perceived_phones)
    if keep_artificial_sil:
        # when we preserve the artificial sils, the canonical phones and
        # perceived phones should be perfectly aligned
        assert len(canonical_phones) == len(perceived_phones)
    if rm_repetitive_sil:
        canonical_phones = remove_repetitive_sil(canonical_phones)
        perceived_phones = remove_repetitive_sil(perceived_phones)
    return " ".join(canonical_phones), " ".join(perceived_phones)

def tier_to_list(tier):
    return [interval.mark for interval in tier]

def get_word_bounds(word_tier, phone_tier):
    """
    word_tier: [(minTime, maxTime, word1), (minTime, maxTime, word2), ...]
    phone_tier: [(minTime, maxTime, phn1), (minTime, maxTime, phn2), ...]
    return word_bounds: [(0, 3), (4, 7), ...], length should be the same as word_tier
    """
    phn_interval_list = [x for x in phone_tier]
    word_interval_list = [x for x in word_tier]
    phn_idx = 0
    word_bounds = []
    for word_idx in range(len(word_interval_list)):
        word_interval = word_interval_list[word_idx]
        bound = []
        while word_interval.maxTime >= phn_interval_list[phn_idx].maxTime:
            bound.append(phn_idx)
            phn_idx += 1
            if phn_idx == len(phn_interval_list):
                break
        word_bounds.append(bound)

    word_bounds = [(x[0], x[-1]) for x in word_bounds if x != []]
    return word_bounds


def remove_repetitive_sil(phone_list):
    # Filtering out consecutive silences by applying a mask with `True` marking
    # which sils to remove
    # e.g.
    # phone_list          [  "a", "sil", "sil",  "sil",   "b"]
    # ---
    # create:
    # remove_sil_mask   [False,  True,  True,  False,  False]
    # ---
    # so end result is:
    # phone_list ["a", "sil", "b"]

    remove_sil_mask = [True if x == "sil" else False for x in phone_list]

    for i, val in enumerate(remove_sil_mask):
        if val is True:
            if i == len(remove_sil_mask) - 1:
                remove_sil_mask[i] = False
            elif remove_sil_mask[i + 1] is False:
                remove_sil_mask[i] = False

    phone_list = [
        phon for i, phon in enumerate(phone_list) if not remove_sil_mask[i]
    ]
    return phone_list

def normalize_tier_mark(tier: IntervalTier,
                        mode="NormalizePhoneCanonical", keep_artificial_sil=False) -> IntervalTier:
    """Normalize the marks of an IntervalTier.
    Refer to the code for supported modes.
    Args:
        tier: An IntervalTier object.
        mode: The filter function for each mark in the tier.
    Returns:
        tier: Mark-normalized tier.
    """
    tier = copy.deepcopy(tier)
    tier_out = IntervalTier()
    if mode not in {"NormalizePhoneCanonical",
                    "NormalizePhonePerceived",
                    "NormalizePhoneAnnotation",
                    "NormalizeWord"}:
        raise ValueError("Mode %s is not valid.", mode)
    for each_interval in tier.intervals:
        if mode == "NormalizePhoneCanonical":
            # Only keep the canonical pronunciation.
            p = normalize_phone(each_interval.mark, True, True, keep_artificial_sil)
        elif mode == "NormalizePhonePerceived":
            # Only keep the perceived pronunciation.
            p = normalize_phone(each_interval.mark, True, False, keep_artificial_sil)
        elif mode == "NormalizePhoneAnnotation":
            # Keep the annotations.
            p = normalize_phone(each_interval.mark, False)
        elif mode == "NormalizeWord":
            p = normalize_word(each_interval.mark)

        if p is None:
            continue
        if p == 'ax':
            p = 'ah'
        each_interval.mark = p
        assert p in ARPA_PHONEMES + ["err"], pdb.set_trace()
        tier_out.addInterval(each_interval)
    return tier_out

def normalize_phone(s: str, is_rm_annotation=True, is_phoneme_canonical=True,
                     keep_artificial_sil=False) -> str:
    """Normalize phoneme labels to lower case, stress-free form.
    This will also deal with L2-ARCTIC annotations.
    Args:
        s: A phoneme annotation.
        is_rm_annotation: [optional] Only return the canonical pronunciation if
        set to true, otherwise will keep the annotations.
        is_phoneme_canonical: [optional] If set to true, return canonical phoneme; otherwise
        return perceived phoneme.
        keep_artificial_sil: If true, will keep the artificial sil produced by the way L2ARCTIC was annotated.
                            If false, will not have the sil
                            e.g. when false, 'ah, sil, d' canonical: ah, perceived: None
                                 when true, 'ah, sil, d' canonical: ah, perceived: sil
    Returns:
        Normalized phoneme (canonical pronunciation or with annotations).
    """
    t = s.lower()
    pattern = re.compile(r"[^a-z,]")
    parse_tag = pattern.sub("", t)
    if is_sil(parse_tag):
        return "sil"
    if len(parse_tag) == 0:
        raise ValueError("Input %s is invalid.", s)
    if len(parse_tag.split(",")) == 1:
        if parse_tag.split(",")[0] == 'ax':
            return 'ah'
        else:
            return parse_tag.split(",")[0]
    if is_rm_annotation:
        # This handles the L2-ARCTIC annotations, here we extract the canonical
        # pronunciation
        if keep_artificial_sil:
            if is_phoneme_canonical:
                return parse_tag.split(",")[0]
            else:
                return parse_tag.split(",")[1]
        elif not keep_artificial_sil:
            if is_phoneme_canonical:
                if parse_tag.split(",")[2] in ['s', 'd']:
                    return parse_tag.split(",")[0]
                elif parse_tag.split(",")[2] == 'a':
                    return None
            else:
                if parse_tag.split(",")[2] in ['s', 'a']:
                    return parse_tag.split(",")[1]
                elif parse_tag.split(",")[2] == 'd':
                    return None
    else:
        return parse_tag

def is_sil(s: str) -> bool:
    """Test if the input string represents silence.
    Args:
        s: A phoneme label.
    Returns:
        True if is silence, otherwise False.
    """
    if s.lower() in {"sil", "sp", "spn", "pau", ""}:
        return True
    else:
        return False


if __name__ == "__main__":

    prepare_l2arctic(
    data_folder=sys.argv[1],
    save_json_train="data/train.json",
    save_json_test="data/test.json",
    metadata_l2arctic="data/metadata_l2arctic",
    test_spks=['TLV', 'NJS', 'TNI', 'TXHC', 'ZHAA', 'YKWK',]
)


