import os
import sys
import json
import argparse
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.dataio.wer import print_alignments, _print_alignment
from speechbrain.utils.metric_stats import MetricStats, ErrorRateStats

EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}

class MpdStats(MetricStats):
    """Compute MDD eval metrics, adapted from speechbrain.utils.metric_stats.MetricStats
    see speechbrain.utils.metric_stats.MetricStats
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict,
        canonical,
        perceived,
        predict_len=None,
        canonical_len=None,
        perceived_len=None,
        ind2lab=None,
    ):
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if canonical_len is not None:
            canonical = undo_padding(canonical, canonical_len)
        if perceived_len is not None:
            perceived = undo_padding(perceived, perceived_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            canonical = ind2lab(canonical)
            perceived = ind2lab(perceived)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        ## remove parallel sil in cano and perc
        canonical, perceived = rm_parallel_sil_batch(canonical, perceived)
        assert len(canonical) == len(perceived)  # make sure cano and perc are aligned

        ## remove all sil in hyp
        predict = [[x for x in y if x!= "sil"] for y in predict]


        alignments = [extract_alignment(c, p) for c, p in zip(canonical, perceived)]
        wer_details = wer_details_for_batch(ids=ids,
                                           refs=[[s for s in c if s != "sil"] for c in canonical],
                                           hyps=predict,
                                           compute_alignments=True)
        ## let's be clear about the two alignments' names, rename the keys
        for a, p, det in zip(alignments, perceived, wer_details):
            det["alignment_cano2hyp"] = det.pop("alignment")
            det["canonical"] = det.pop("ref_tokens")
            det["hypothesis"] = det.pop("hyp_tokens")
            det.update({"alignment_cano2perc": a})
            det.update({"perceived": [s for s in p if s != "sil"]})


        self.scores.extend(wer_details)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.
        * See MetricStats.summarize()
        """
        # self.summary = wer_summary(self.scores)
        self.summary = mpd_summary(self.scores)

        # Add additional, more generic key
        self.summary["mpd_f1"] = self.summary["f1"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_mpd_details(self.scores, self.summary, filestream)


def mpd_eval_on_dataset(in_json, mpd_file=sys.stdout, per_file=None):

    if per_file:
        error_rate_stats = ErrorRateStats()
    total_wer_details = []

    for wav_id, wav_data in in_json.items():
        cano_phns = wav_data["canonical_phn"].split()
        perc_phns = wav_data["phn"].split()
        cano_phns, perc_phns = rm_parallel_sil(cano_phns, perc_phns)
        assert len(cano_phns) == len(perc_phns)

        alignment = extract_alignment(cano_phns, perc_phns)


        hyp = [s for s in wav_data["hyp"].split() if s!= "sil"]
        # hyp = wav_data["hyp"].split()
        wer_details = wer_details_for_batch(ids=[wav_id],
                                           refs=[[s for s in cano_phns if s != "sil"]],
                                           hyps=[hyp],
                                           compute_alignments=True)[0]
        ## let's be clear about the two alignments' names, rename the keys
        wer_details["alignment_cano2hyp"] = wer_details.pop("alignment")
        wer_details["canonical"] = wer_details.pop("ref_tokens")
        wer_details["hypothesis"] = wer_details.pop("hyp_tokens")
        wer_details.update({"alignment_cano2perc": alignment})
        wer_details.update({"perceived": [s for s in perc_phns if s != "sil"]})
        wer_details.update({"wav_id": wav_id})

        total_wer_details.append(wer_details)


        if per_file:
            error_rate_stats.append(ids=[wav_id],
                                    target=[[s for s in cano_phns if s != "sil"]],
                                    predict=[hyp])

    if per_file:
        error_rate_stats.write_stats(per_file)
    mpd_stats = mpd_summary(total_wer_details)
    print_mpd_details(total_wer_details, mpd_stats, mpd_file)


def mpd_summary(total_wer_details):

    total_ta, total_fr, total_fa, total_tr, total_cor_diag, total_err_diag = 0, 0, 0, 0, 0, 0
    total_ins, total_del, total_sub, total_eq = 0, 0, 0, 0
    for det in total_wer_details:

        total_ins += len([a for a in det["alignment_cano2perc"] if a[0] == "I"])
        total_del += len([a for a in det["alignment_cano2perc"] if a[0] == "D"])
        total_sub += len([a for a in det["alignment_cano2perc"] if a[0] == "S"])
        total_eq += len([a for a in det["alignment_cano2perc"] if a[0] == "="])


        ta, fr, fa, tr, cor_diag, err_diag = mpd_stats(det["alignment_cano2perc"],
                                                       det["alignment_cano2hyp"],
                                                       det["canonical"],
                                                       det["perceived"],
                                                       det["hypothesis"])
        assert tr == (cor_diag + err_diag)
        det.update({
                      "ta": ta,
                      "fr": fr,
                      "fa": fa,
                      "tr": tr,
                      "cor_diag": cor_diag,
                      "err_diag": err_diag,
                    })

        total_ta += ta
        total_fr += fr
        total_fa += fa
        total_tr += tr
        total_cor_diag += cor_diag
        total_err_diag += err_diag

    precision = 1.0*total_tr / (total_fr + total_tr)
    recall = 1.0*total_tr / (total_fa + total_tr)
    f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "total_eq": total_eq,
        "total_sub": total_sub,
        "total_del": total_del,
        "total_ins": total_ins,
        "ta": total_ta,
        "fr": total_fr,
        "fa": total_fa,
        "tr": total_tr,
        "cor_diag": total_cor_diag,
        "err_diag": total_err_diag,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def print_mpd_details(wer_details, mpd_stats, mpd_file):


    print("In original annotation: \nTotal Eq: {}, Total Sub: {}, Total Del: {}, Total Ins: {}".format(\
            mpd_stats["total_eq"], mpd_stats["total_sub"], mpd_stats["total_del"], mpd_stats["total_ins"]), file=mpd_file)
    print("Overall MPD results: \nTrue Accept: {}, False Rejection: {}, False Accept: {}, True Rejection: {}, Corr Diag: {}, Err Diag: {}".format(\
            mpd_stats["ta"], mpd_stats["fr"], mpd_stats["fa"], mpd_stats["tr"], mpd_stats["cor_diag"], mpd_stats["err_diag"]), file=mpd_file)
    print("Precision: {}, Recall: {}, F1: {}".format(mpd_stats["precision"], mpd_stats["recall"], mpd_stats["f1"]), file=mpd_file)

    for det in wer_details:
        print("="*80, file=mpd_file)
        print(det["key"], file=mpd_file)
        print("Human annotation: Canonical vs Perceived:", file=mpd_file)
        _print_alignment(alignment=det["alignment_cano2perc"],
                         a=det["canonical"],
                         b=det["perceived"],
                         file=mpd_file)

        print("Model Prediction: Canonical vs Hypothesis:", file=mpd_file)
        _print_alignment(alignment=det["alignment_cano2hyp"],
                         a=det["canonical"],
                         b=det["hypothesis"],
                         file=mpd_file)
        print("True Accept: {}, False Rejection: {}, False Accept: {}, True Reject: {}, Corr Diag: {}, Err Diag: {}".format(\
                det["ta"], det["fr"], det["fa"], det["tr"], det["cor_diag"], det["err_diag"]), file=mpd_file)



def mpd_stats(align_c2p, align_c2h, c, p, h):
    """
    schema: [(operator, idx_i(None), idx_j(None))]
    c: canonical
    p: perceived
    h: hypothesis
    """
    cnt = 0
    ta, fr, fa, tr, cor_diag, err_diag = 0, 0, 0, 0, 0, 0
    # cano_len = 1 + max(x[1] for x in align_c2p)
    assert max(x[1] for x in align_c2p if x[1] is not None) ==  max(x[1] for x in align_c2h if x[1] if not None)

    i, j = 0, 0
    while i < len(align_c2p) and j < len(align_c2h):
        ## sub and del cases
        if align_c2p[i][1] is not None and \
           align_c2h[j][1] is not None and \
           align_c2p[i][1] == align_c2h[j][1]:
            assert align_c2p[i][0] != EDIT_SYMBOLS["ins"]
            assert align_c2h[j][0] != EDIT_SYMBOLS["ins"]
            if align_c2p[i][0] == EDIT_SYMBOLS["eq"]:
                ## canonical cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    ta += 1
                else:
                    fr += 1
            elif align_c2p[i][0] != EDIT_SYMBOLS["eq"]:
                ## mispronunciation cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    fa += 1
                else:
                    tr += 1
                    if align_c2p[i][0] != align_c2h[j][0]:
                        err_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["del"] and align_c2h[j][0] == EDIT_SYMBOLS["del"]:
                        cor_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["sub"] and align_c2h[j][0] == EDIT_SYMBOLS["sub"]:
                        if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                            cor_diag += 1
                        else:
                            err_diag += 1
            i += 1
            j += 1
        ## ins cases
        elif align_c2p[i][1] is None and \
             align_c2h[j][1] is not None:
            fa += 1
            i += 1
        elif align_c2p[i][1] is not None and  \
             align_c2h[j][1] is None:
            fr += 1
            j += 1
        elif align_c2p[i][1] is None and align_c2h[j][1] is None:
            tr += 1
            if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                cor_diag += 1
            else:
                err_diag += 1
            i += 1
            j += 1
    if i == len(align_c2p) and j != len(align_c2h):
        fr += len(align_c2h[j:])
    if i != len(align_c2p) and j == len(align_c2h):
        fa += len(align_c2p[j:])

    return ta, fr, fa, tr, cor_diag, err_diag


def extract_alignment(a, b, gap_token="sil"):
    """
    a, b are two aligned lists (i.e. same length)
    gap_token is the artificial token placeholder used in L2Arctic annotation. In this case is a `sil` token
    """
    alignment = []
    idx_a, idx_b = 0, 0
    for str_a, str_b in zip(a, b):
        if str_a == gap_token and str_b != gap_token:
            alignment.append((EDIT_SYMBOLS["ins"], None, idx_b))
            idx_b += 1
        elif str_a != gap_token and str_b == gap_token:
            alignment.append((EDIT_SYMBOLS["del"], idx_a, None))
            idx_a += 1
        elif str_a != gap_token and str_b != gap_token and str_a != str_b:
            alignment.append((EDIT_SYMBOLS["sub"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
        else:
            alignment.append((EDIT_SYMBOLS["eq"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
    return alignment

def rm_parallel_sil_batch(canos, percs):
    canos_out, percs_out = [], []
    assert len(canos) == len(percs)  ## batch size
    for cano, perc in zip(canos, percs):
        cano, perc = rm_parallel_sil(cano, perc)
        canos_out.append(cano)
        percs_out.append(perc)
    return canos_out, percs_out

def rm_parallel_sil(canos, percs):
    canos_out, percs_out = [], []
    assert len(canos) == len(percs)  ## aligned
    for cano, perc in zip(canos, percs):
        if (cano==perc and cano=="sil"):
            continue
        canos_out.append(cano)
        percs_out.append(perc)
    return canos_out, percs_out


def main(args):
    with open(args.json_path, "r") as f:
        json_data = json.load(f)
    per_file = open(args.per_file, "w")
    mpd_file = open(args.mpd_file, "w")
    mpd_eval_on_dataset(json_data, mpd_file, per_file)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", type=str)
    p.add_argument("--per_file", type=str, default=None)
    p.add_argument("--mpd_file", type=str, default=None)
    args = p.parse_args()

    main(args)
