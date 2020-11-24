# OpenNMT-APE: Fork of OpenNMT that implements the model described in "A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning"

Note: This repository is no longer being maintained and the packages it uses are now very outdated (particularly Hugging Face's `pytorch-pretrained-BERT`, which is now called [transformers](https://github.com/huggingface/transformers). Pull requests to update this repository are very welcome!

## Abstract

Automatic post-editing (APE) seeks to automatically refine the output
of a black-box machine translation (MT) system through human
post-edits. APE systems are usually trained by complementing human
post-edited data with large, artificial data generated through back-
translations, a time-consuming process often no easier than training
a MT system from scratch. In this paper, we propose an alternative
where we fine-tune pre-trained BERT models on both the encoder and
decoder of an APE system, exploring several parameter sharing
strategies. By only training on a dataset of 23K sentences for 3
hours on a single GPU we obtain results that are competitive with
systems that were trained on 5M artificial sentences. When we add
this artificial data, our method obtains state-of-the-art results.

## Preprocess

To preprocess your data so you can train an APE model with our implementation, you need to do the following steps:

- Have a triplet of files to train your APE system (`src`, `mt`, `pe`).
- Use a simple tokenizer on your files. We used [Moses](https://github.com/moses-smt/mosesdecoder) tokenizer (found in `mosesdecoder/scripts/tokenizer/tokenizer.perl`) with the flag `-no-escape`.
- Join the tokenized `.src` and `.mt` files in the same file `.srcmt`, separated by " \[SEP\] ". You can easily do this by running the following command: `pr -tmJ -S" [SEP] " train_data.tok.src train_data.tok.mt > train_data.tok.srcmt`.
- Do the OpenNMT-py preprocess pipeline by running `python preprocess.py -config preprocessing.yml` where the `preprocessing.yml` file is like the following:

```
train_src: train_data.tok.srcmt
train_tgt: train_data.tok.pe

valid_src: dev.tok.srcmt
valid_tgt: dev.tok.pe

save_data: prep-data

src_vocab_size: 200000
tgt_vocab_size: 200000

shard_size: 100000

bert_src: bert-base-multilingual-cased
bert_tgt: bert-base-multilingual-cased

src_seq_length: 200
tgt_seq_length: 100
```

## Train

To train, run `python train.py -config train-config.yml` where `train-config.yml` is:

```
save_model: ape-model

data: prep-data
train_steps: 50000
start_decay_steps: 50000
valid_steps: 1000
save_checkpoint_steps: 1000
keep_checkpoint: 30

# Dimensionality
rnn_size: 768 #!
word_vec_size: 768 #!
transformer_ff: 3072 #!
heads: 12 #!
layers: 12 #!

# Embeddings
position_encoding: 'true' #!
share_embeddings: 'true' #!
share_decoder_embeddings: 'true' #!

# Encoder
encoder_type: bert #!
enc_bert_type: bert-base-multilingual-cased #!

# Decoder
decoder_type: bert #!
dec_bert_type: bert-base-multilingual-cased #!
bert_decoder_token_type: B #!

# Layer Sharing
bert_decoder_init_context: 'true'
share_self_attn: 'true'
# tie_context_attn: 'true'
# share_feed_forward: 'true'

# Regularization
dropout: 0.1
label_smoothing: 0.1

# Optimization
optim: bertadam #!
learning_rate: 0.00005
warmup_steps: 5000
batch_type: tokens
normalization: tokens
accum_count: 2
batch_size: 512
max_grad_norm: 0
param_init: 0
param_init_glorot: 'true'
valid_batch_size: 8

average_decay: 0.0001

# GPU
seed: 42
world_size: 1
gpu_ranks: 0
```

In order for the training to work, parameters shown with a `#!` in front of them are important to stay as shown in the config file above, while others can be finetuned.

## Citation

```
@inproceedings{Correia2019,
author = {Correia, Gon{\c{c}}alo M. and Martins, Andr{\'{e}} F. T.},
booktitle = {Proceedings of ACL},
title = {{A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning}},
year = {2019}
}
```
