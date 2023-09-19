import os
import sys
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import librosa
from tqdm import tqdm
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
        ids = batch.id
        wavs, wav_lens = batch.sig

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
        # Note: sb.decoders.ctc_greedy_decode will also remove padded tokens
        # that is, it return a list of list with different lengths
        sequence = sb.decoders.ctc_greedy_decode(
            p_ctc, wav_lens, blank_id=self.hparams.blank_index
        )
        transcriptions = [" ".join(self.label_encoder.decode_ndim(s)) for s in sequence]


        return ids, transcriptions

    def transcribe_dataset(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest WER
            loader_kwargs # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it.
        if not isinstance(dataset, torch.utils.data.DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)
        self.modules.wav2vec2.model.config.apply_spec_augment = False  # make sure no spec aug applied on wav2vec2

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            wav_ids = []
            transcripts = []
            for batch in tqdm(dataset, dynamic_ncols=True):

                ids, preds = self.compute_forward(batch, stage=sb.Stage.TEST)

                transcripts.extend(preds)
                wav_ids.extend(ids)

        return wav_ids, transcripts


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:

    inference_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["inference_annotation"],
        replacements={"data_root": data_folder},
    )
    inference_data = inference_data.filtered_sorted(sort_key="duration")

    datasets = [inference_data]
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


    # 3. Fit encoder:
    # Load the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load(lab_enc_file)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig"],
    )

    return inference_data, label_encoder


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
    inference_data, label_encoder = dataio_prep(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    wav_ids, transcripts = asr_brain.transcribe_dataset(
        dataset=inference_data, # Must be obtained from the dataio_function
        min_key="PER", # We load the model with the lowest PER
        loader_kwargs=hparams["inference_dataloader_opts"], # opts for the dataloading
    )
    with open(hparams["inference_annotation"], "r") as json_f:
        unlabeled_data = json.load(json_f)
    for wav_id, transcript in zip(wav_ids, transcripts):
        unlabeled_data[wav_id].update({"pred_phns": transcript})

    ## save as new json file
    with open(hparams["inference_annotation_saved"], "w") as json_f_save:
        json.dump(unlabeled_data, json_f_save, indent=2)
