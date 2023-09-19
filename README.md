# mpl-mdd

Code for our paper "[Improving Mispronunciation Detection with Wav2vec2-based Momentum Pseudo-Labeling for Accentedness and Intelligibility Assessment](https://arxiv.org/abs/2203.15937)". An audio demo is available [here](https://mu-y.github.io/speech_samples/mdd_IS22/). 

This repo contains code for fine-tuning a wav2vec2-based MDD model with momentum pseudo-labeling (MPL). The implementation is based on [SpeechBrain](https://github.com/speechbrain/speechbrain).

## Pull the repo
```
git clone git@github.com:Mu-Y/mpl-mdd.git
cd mpl-mdd
git submodule update --init --recursive
```

## Install dependencies and set up env
Install the requirements by SpeechBrain and some extras.
```
cd mpl-mdd/speechbrain
pip install -r requirements.txt
pip install textgrid transformers librosa
```
Append the path to speechbrain module to `PYTHONPATH`.
```
export PYTHONPATH=$PYTHONPATH:<path to mpl-mdd/speechbrain>
```

## Data preperation
First, download [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) dataset, and unzip it. Then run the following commands:
```
# for labeled samples - get train.json and test.json
python l2arctic_prepare.py <path to L2-ARCTIC>

# for unlabled samples - get train_unlabeled.json
python l2arctic_unlabeled_prepare.py <path to L2-ARCTIC>

# split dev set from training - get train-train.json and train-dev.json
python split_train_dev.py --in_json=data/train.json --out_json_train=data/train-train.json --out_json_dev=data/train-dev.json
```



## Training
### Step 1
Fine-tune a pre-trained wav2vec2 model on labeled samples.
```
python train.py hparams/train.yaml
```
### Step 2
Fine-tune the model from step 1 with momentum pseudo-labeling, using both labeled and unlabled samples.
```
python train_mpl.py hparams/train_mpl.yaml
```

## Evaluate the trained model
```
python evaluate.py hparams/evaluate.yaml
```
This will print PER and MDD F1, and write the PER and MDD details files. Note that the F1 printed here is from a MDD evaluator that is quite different from the one we used in the paper. The one used in the paper follows the prior work here: https://github.com/cageyoko/CTC-Attention-Mispronunciation. You need to convert the predictions into the acceptable format of that evaluator, which should be very straightforward.

## Inference with the trained model
```
python transcribe.py hparams/transcribe.yaml
```
By default, this command will write predictions of L2-ARCTIC test set into a json file. You can change the save path in `hparams/transcribe.yaml`.

## Acknowledgements
The code is adapted from several SpeechBrain recipes:
https://github.com/speechbrain/speechbrain/tree/develop/recipes/TIMIT/ASR/seq2seq
https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer

## Citation
```
@inproceedings{yang22IS_Improving,
  author={Mu Yang and Kevin Hirschi and Stephen Daniel Looney and Okim Kang and John H.L. Hansen},
  title={{Improving Mispronunciation Detection with Wav2vec2-based Momentum Pseudo-Labeling for Accentedness and Intelligibility Assessment}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4481--4485},
  doi={10.21437/Interspeech.2022-11039}
}
```
