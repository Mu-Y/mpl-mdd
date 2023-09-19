import os
import sys
import time
from tqdm.contrib import tqdm
import torch
import logging
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.core import Stage
from speechbrain.utils.distributed import run_on_main
from mpd_eval_v3 import MpdStats
import librosa
import json
import itertools
import math

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

            # Note: the None in the function arguments mean that we do not specify predict_len
            # meaning the padding has been removed so we do not need to predicted lengths
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
        # self.seq_metrics = self.hparams.seq_stats()
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
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
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


    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default implementation depends on a few methods being defined
        with a particular behavior:

        * ``compute_forward()``
        * ``compute_objectives()``

        Also depends on having optimizers passed at initialization.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.

        Returns
        -------
        detached loss
        """
        # Managing automatic mixed precision
        if self.auto_mix_prec:

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            if not hasattr(batch, "phn_encoded_target"):
                ## unlabeled batch inference for pseudo labels
                ## make sure modules are in eval() mode, augmentations are DISABLED, see infer_batch()
                pls, pl_lens = self.infer_batch(batch)
                ## then, set to train() mode and ENABLE all augmentations to train on PLs
                self.modules.train()
                if self.hparams.wav2vec2_specaug:
                    self.modules.wav2vec2.model.config.apply_spec_augment = True
                ## forward pass on unlabled batch, with all augmentations ENABLED
                outputs_u = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives_unlabeled(outputs_u, pls, pl_lens)
            elif hasattr(batch, "phn_encoded_target"):
                ## labeled batch - make sure modules are in train() mode, augmentations are added
                self.modules.train()
                if self.hparams.wav2vec2_specaug:
                    self.modules.wav2vec2.model.config.apply_spec_augment = True
                outputs_l = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs_l, batch, sb.Stage.TRAIN)

            # normalize the loss by gradient_accumulation step
            (loss / self.hparams.gradient_accumulation).backward()

            if self.step % self.hparams.gradient_accumulation == 0:
                # gradient clipping & early stop if loss is not fini
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.adam_optimizer.step()

                self.wav2vec_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()

                # momentum update on teacher model
                self.teacher_momentum_update()

        return loss.detach().cpu()

    def infer_batch(self, batch):
        ## make sure modules are in eval() mode, augmentations are disabled
        self.modules_teacher.eval()
        self.modules_teacher.wav2vec2.model.config.apply_spec_augment = False
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        with torch.no_grad():
            # some wav2vec models (e.g. large-lv60) needs attention_mask
            if self.modules_teacher.wav2vec2.feature_extractor.return_attention_mask:
                attn_mask = make_attn_mask(wavs, wav_lens)
            else:
                attn_mask = None
            feats = self.modules_teacher.wav2vec2(wavs, attention_mask=attn_mask)
            x = self.modules_teacher.enc(feats)

            # output layer for ctc log-probabilities
            logits = self.modules_teacher.ctc_lin(x)
            p_ctc = self.hparams.log_softmax(logits)

            pseudo_labels = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
            max_len = max(len(x) for x in pseudo_labels)
            pseudo_label_lens = torch.tensor([float(len(x)/max_len) for x in pseudo_labels])
            pseudo_labels = pad_sequence(
                [torch.tensor(x) for x in pseudo_labels],
                batch_first=True
            )
        return pseudo_labels.to(self.device), pseudo_label_lens.to(self.device)

    def compute_objectives_unlabeled(self, predictions, targets, target_lens):
        "Simply compute the CTC loss"

        p_ctc, wav_lens = predictions

        loss_ctc = self.hparams.ctc_cost(p_ctc, targets, wav_lens, target_lens)
        return loss_ctc

    def teacher_momentum_update(self):
        with torch.no_grad():
            for pname, param in self.modules_teacher.state_dict().items():
                param = self.momentum_factor * param + (1 - self.momentum_factor) * self.modules.state_dict()[pname]


    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.model.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        ## NOTE: make sure to use the "best" model to continual training
        ## so we set the `min_key` argument
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device),
                min_key="PER"
            )

        ## set the epoch_counter to start from epoch 50
        self.hparams.epoch_counter.current=50


        ## initialize teacher model - load from the same base model ckpt
        chosen_ckpt = self.checkpointer.find_checkpoint(min_key="PER")
        model_layers = self.hparams.model_teacher
        wav2vec2_layers = self.hparams.wav2vec2_teacher
        model_layers.load_state_dict(
            torch.load(chosen_ckpt.paramfiles["model"], map_location=torch.device(self.device))
        )
        wav2vec2_layers.load_state_dict(
            torch.load(chosen_ckpt.paramfiles["wav2vec2"], map_location=torch.device(self.device))
        )
        self.modules_teacher.eval()

        self.set_momentum_factor(
            self.n_train_batch,
            self.hparams.epoch_counter.limit-self.hparams.epoch_counter.current
        )

    def set_momentum_factor(self, n_train_batch, n_epochs):
        total_steps = float(n_train_batch * n_epochs // self.hparams.gradient_accumulation)
        self.momentum_factor = math.exp( (1/total_steps) * math.log(self.hparams.base_model_factor))
        logger.info("Momentum Factor: {}".format(self.momentum_factor))

    def fit(
        self,
        epoch_counter,
        train_data_l,
        train_data_u,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate epochs and datasets to improve objective.
        Relies on the existence of multiple functions that can (or should) be
        overridden. The following methods are used and expected to have a
        certain behavior:
        * ``fit_batch()``
        * ``evaluate_batch()``
        * ``update_average()``
        If the initialization was done with distributed_count > 0 and the
        distributed_backend is ddp, this will generally handle multiprocess
        logic, like splitting the training data into subsets for each device and
        only saving a checkpoint on the main process.
        Arguments
        ---------
        epoch_counter : iterable
            Each call should return an integer indicating the epoch count.
        train_set : Dataset, DataLoader
            A set of data to use for training. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        valid_set : Dataset, DataLoader
            A set of data to use for validation. If a Dataset is given, a
            DataLoader is automatically created. If a DataLoader is given, it is
            used directly.
        train_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the train_loader
            (if train_set is a Dataset, not DataLoader).
            E.G. batch_size, num_workers.
            DataLoader kwargs are all valid.
        valid_loader_kwargs : dict
            Kwargs passed to `make_dataloader()` for making the valid_loader
            (if valid_set is a Dataset, not DataLoader).
            E.g., batch_size, num_workers.
            DataLoader kwargs are all valid.
        progressbar : bool
            Whether to display the progress of each epoch in a progressbar.
        """

        self.n_train_batch = len(train_data_l) + len(train_data_u)

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Iterate epochs
        for epoch in epoch_counter:

            ## chain labeled data loader and unlabeled data loader
            train_set = itertools.chain(train_data_l, train_data_u)

            # Training stage
            self.on_stage_start(Stage.TRAIN, epoch)
            self.modules.train()

            # Reset nonfinite count to 0 each epoch
            self.nonfinite_count = 0

            if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"
            ):
                self.train_sampler.set_epoch(epoch)

            # Time since last intra-epoch checkpoint
            last_ckpt_time = time.time()

            # Only show progressbar if requested and main_process
            enable = progressbar and sb.utils.distributed.if_main_process()
            with tqdm(
                train_set,
                initial=self.step,
                total=self.n_train_batch,
                dynamic_ncols=True,
                disable=not enable,
            ) as t:
                for batch in t:
                    self.step += 1
                    loss = self.fit_batch(batch)
                    self.avg_train_loss = self.update_average(
                        loss, self.avg_train_loss
                    )
                    t.set_postfix(train_loss=self.avg_train_loss)

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                    if (
                        self.checkpointer is not None
                        and self.ckpt_interval_minutes > 0
                        and time.time() - last_ckpt_time
                        >= self.ckpt_interval_minutes * 60.0
                    ):
                        # This should not use run_on_main, because that
                        # includes a DDP barrier. That eventually leads to a
                        # crash when the processes'
                        # time.time() - last_ckpt_time differ and some
                        # processes enter this block while others don't,
                        # missing the barrier.
                        if sb.utils.distributed.if_main_process():
                            self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()

            # Run train "on_stage_end" on all processes
            self.on_stage_end(Stage.TRAIN, self.avg_train_loss, epoch)
            self.avg_train_loss = 0.0
            self.step = 0

            # Validation stage
            if valid_set is not None:
                self.on_stage_start(Stage.VALID, epoch)
                self.modules.eval()
                avg_valid_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        valid_set, dynamic_ncols=True, disable=not enable
                    ):
                        self.step += 1
                        loss = self.evaluate_batch(batch, stage=Stage.VALID)
                        avg_valid_loss = self.update_average(
                            loss, avg_valid_loss
                        )

                        # Debug mode only runs a few batches
                        if self.debug and self.step == self.debug_batches:
                            break

                    # Only run validation "on_stage_end" on main process
                    self.step = 0
                    run_on_main(
                        self.on_stage_end,
                        args=[Stage.VALID, avg_valid_loss, epoch],
                    )

            # Debug mode only runs a few epochs
            if self.debug and epoch == self.debug_epochs:
                break


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    ## labeled training data
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    ## unlabled training data
    train_data_u = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["unlabeled_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, train_data_u, valid_data, test_data]
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
    @sb.utils.data_pipeline.takes("perceived_train_target")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
    )
    def text_pipeline_train(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded_list = label_encoder.encode_sequence(phn_list)
        yield phn_encoded_list
        phn_encoded = torch.LongTensor(phn_encoded_list)
        yield phn_encoded

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

    sb.dataio.dataset.add_dynamic_item([train_data], text_pipeline_train)
    sb.dataio.dataset.add_dynamic_item([valid_data, test_data], text_pipeline_test)

    # 3. Fit encoder:
    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list_target",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        [train_data],
        ["id", "sig", "phn_encoded_target"],
    )
    sb.dataio.dataset.set_output_keys(
        [train_data_u],
        ["id", "sig"],
    )
    sb.dataio.dataset.set_output_keys(
        [valid_data, test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived"],
    )

    return train_data, train_data_u, valid_data, test_data, label_encoder


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
    train_data_l, train_data_u, valid_data, test_data, label_encoder = dataio_prep(hparams)

    ## build data loaders for all datasets
    train_data_l = DataLoader(
        train_data_l,
        batch_size=hparams["batch_size_labeled"],
        drop_last=False,
        shuffle=True,
        sampler=None,
        collate_fn=PaddedBatch,
        num_workers=hparams["num_workers"]
    )
    train_data_u = DataLoader(
        train_data_u,
        batch_size=hparams["batch_size_unlabeled"],
        drop_last=False,
        shuffle=True,
        sampler=None,
        collate_fn=PaddedBatch,
        num_workers=hparams["num_workers"]
    )
    valid_data = DataLoader(
        valid_data,
        batch_size=hparams["batch_size_labeled"],
        drop_last=False,
        shuffle=False,
        sampler=None,
        collate_fn=PaddedBatch,
        num_workers=hparams["num_workers"]
    )
    test_data = DataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        sampler=None,
        collate_fn=PaddedBatch,
        num_workers=1
    )


    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    asr_brain.modules_teacher = torch.nn.ModuleDict(hparams["modules_teacher"]).to(asr_brain.device)

    # Training/validation loop
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data_l,
        train_data_u,
        valid_data,
        train_loader_kwargs=None,
        valid_loader_kwargs=None,
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=None,
    )
