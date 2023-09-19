import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from mpd_eval_v3 import MpdStats
import librosa
import json

logger = logging.getLogger(__name__)

def make_attn_mask(wavs, wav_lens):
    """
    wav_lens: relative lengths(i.e. 0-1) of a batch. shape: (bs, )
    return a tensor of shape (bs, seq_len), representing mask on allowed positions.
            1 for regular tokens, 0 for padded tokens
    """
    abs_lens = (wav_lens*wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # phns_bos, _ = batch.phn_encoded_bos

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # some wav2vec models (e.g. large-lv60) needs attention_mask
        if self.modules.wav2vec2.feature_extractor.return_attention_mask:
            attn_mask = make_attn_mask(wavs, wav_lens)
        else:
            attn_mask = None
        feats = self.modules.wav2vec2(wavs, attention_mask=attn_mask)
        x = self.modules.enc(feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        return p_ctc, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."

        p_ctc, wav_lens = predictions

        ids = batch.id
        # phns_eos, phn_lens_eos = batch.phn_encoded_eos
        targets, target_lens = batch.phn_encoded_target
        if stage != sb.Stage.TRAIN:
            canonicals, canonical_lens = batch.phn_encoded_canonical
            perceiveds, perceived_lens = batch.phn_encoded_perceived

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        loss = loss_ctc

        # Record losses for posterity
        if stage != sb.Stage.TRAIN:
            # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
            # that is, it return a list of list with different lengths
            sequence = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            self.ctc_metrics.append(ids, p_ctc, targets, wav_lens, target_lens)

            self.per_metrics.append(
                ids=ids,
                predict=sequence,
                target=targets,
                predict_len=None,
                target_len=target_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )
            self.mpd_metrics.append(
                ids=ids,
                predict=sequence,
                canonical=canonicals,
                perceived=perceiveds,
                predict_len=None,
                canonical_len=canonical_lens,
                perceived_len=perceived_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.ctc_metrics = self.hparams.ctc_stats()
        if self.hparams.wav2vec2_specaug:
            self.modules.wav2vec2.model.config.apply_spec_augment = True

        if stage != sb.Stage.TRAIN:
            self.modules.wav2vec2.model.config.apply_spec_augment = False
            self.per_metrics = self.hparams.per_stats()
            self.mpd_metrics = MpdStats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            per = self.per_metrics.summarize("error_rate")
            mpd_f1 = self.mpd_metrics.summarize("mpd_f1")

        if stage == sb.Stage.VALID:

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_adam": self.adam_optimizer.param_groups[0]["lr"],
                    "lr_wav2vec": self.wav2vec_optimizer.param_groups[0]["lr"],
                },
                train_stats={"loss": self.train_loss},
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "PER": per,
                    "mpd_f1": mpd_f1
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"PER": per, "mpd_f1": mpd_f1}, min_keys=["PER"], max_keys=["mpd_f1"]
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": 0},
                test_stats={"loss": stage_loss, "PER": per, "mpd_f1": mpd_f1},
            )
            with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nPER stats:\n")
                self.per_metrics.write_stats(w)
                print(
                    "CTC and PER stats written to file",
                    self.hparams.wer_file,
                )
            with open(self.hparams.mpd_file, "w") as m:
                m.write("MPD results and stats:\n")
                self.mpd_metrics.write_stats(m)
                print(
                    "MPD results and stats written to file",
                    self.hparams.mpd_file,
                )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        # # sample rate change to 16000, e,g, using librosa
        # sig = torch.Tensor(librosa.core.load(wav, hparams["sample_rate"])[0])
        # Use wav2vec processor to do normalization
        sig = hparams["wav2vec2"].feature_extractor(
            librosa.core.load(wav, hparams["sample_rate"])[0],
            sampling_rate=hparams["sample_rate"],
        ).input_values[0]
        sig = torch.Tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
    )
    def text_pipeline_test(target, canonical, perceived):
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_canonical = canonical.strip().split()
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived

    sb.dataio.dataset.add_dynamic_item([test_data], text_pipeline_test)

    # 3. Fit encoder:
    # Load the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load(lab_enc_file)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        [test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
    )

    return test_data, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    test_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Test
    asr_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="PER"
    )
