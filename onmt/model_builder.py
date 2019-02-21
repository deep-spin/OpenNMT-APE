"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules

from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token]
                   for _, f in text_field if f.use_vocab]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field if f.use_vocab]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # Build embeddings.
    if model_opt.model_type == "text":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings and model_opt.encoder_type != 'bert':
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            if not model_opt.copy_attn:
                generator[0].weight = decoder.embeddings.word_lut.weight
            else:
                generator.linear.weight = decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility
        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    elif model_opt.encoder_type != 'bert' or model_opt.decoder_type != 'bert':
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if (hasattr(model.decoder, 'embeddings')
                and not model_opt.decoder_type == 'bert'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    if model_opt.encoder_type == 'bert' or model_opt.decoder_type == 'bert':
        if model_opt.bert_type != 'none':
            model_opt.enc_bert_type = model_opt.bert_type
            model_opt.dec_bert_type = model_opt.bert_type

        if model_opt.enc_bert_type != 'none' and checkpoint is None:
                model.encoder.initialize_bert(model_opt.enc_bert_type)

        if model_opt.dec_bert_type != 'none' and checkpoint is None:
                model.decoder.initialize_bert(model_opt.dec_bert_type)

        # Tie word embedding layer of encoder BERT and decoder
        if model_opt.encoder_type == 'bert' and model_opt.share_embeddings:
            decoder.embeddings.word_lut.weight = \
                encoder.embeddings.word_lut.weight

        # Tie decoder word embedding layer with generator weights
        if model_opt.share_decoder_embeddings:
            if not model_opt.copy_attn:
                generator[0].weight = \
                    decoder.embeddings.word_lut.weight
            else:
                generator.linear.weight = \
                    decoder.embeddings.word_lut.weight

    if model_opt.encoder_type == 'bert' and model_opt.decoder_type == 'bert':
        # Tie word, position and token_type embedding
        # layers of encoder and decoder BERT
        if model_opt.share_embeddings:
            decoder.embeddings.position_embeddings.weight = \
                encoder.embeddings.position_embeddings.weight
            decoder.embeddings.token_type_embeddings.weight = \
                encoder.embeddings.token_type_embeddings.weight

        # Tie self-attention between encoder and decoder
        if model_opt.share_self_attn:
            for encoder_layer, decoder_layer in zip(
                    encoder.encoder.layer,
                    decoder.transformer_layers):
                # QUERY
                clone_or_share_layer(
                    decoder_layer.self_attn.linear_query,
                    encoder_layer.attention.self.query,
                    share=True)

                # KEY
                clone_or_share_layer(
                    decoder_layer.self_attn.linear_keys,
                    encoder_layer.attention.self.key,
                    share=True)

                # VALUE
                clone_or_share_layer(
                    decoder_layer.self_attn.linear_values,
                    encoder_layer.attention.self.value,
                    share=True)

                # MULTIHEAD ATTN FINAL LINEAR LAYER
                clone_or_share_layer(
                    decoder_layer.self_attn.final_linear,
                    encoder_layer.attention.output.dense,
                    share=True)

        # Tie context-attention with self-attention
        if model_opt.tie_context_attn:
            for decoder_layer in decoder.transformer_layers:
                # QUERY
                clone_or_share_layer(
                    decoder_layer.context_attn.linear_query,
                    decoder_layer.self_attn.linear_query,
                    share=True)

                # KEY
                clone_or_share_layer(
                    decoder_layer.context_attn.linear_keys,
                    decoder_layer.self_attn.linear_keys,
                    share=True)

                # VALUE
                clone_or_share_layer(
                    decoder_layer.context_attn.linear_values,
                    decoder_layer.self_attn.linear_values,
                    share=True)

                # MULTIHEAD ATTN FINAL LINEAR LAYER
                clone_or_share_layer(
                    decoder_layer.context_attn.final_linear,
                    decoder_layer.self_attn.final_linear,
                    share=True)

        # Tie positionwise feedforward between encoder and decoder
        if model_opt.share_feed_forward:
            for encoder_layer, decoder_layer in zip(
                    encoder.encoder.layer,
                    decoder.transformer_layers):

                # TRANSFORMER FF
                clone_or_share_layer(
                    decoder_layer.intermediate.dense,
                    encoder_layer.intermediate.dense,
                    share=True)

                clone_or_share_layer(
                    decoder_layer.output.dense,
                    encoder_layer.output.dense,
                    share=True)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        model.half()

    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model


def clone_or_share_layer(layer1, layer2, share=False):

    if share:
        layer1.weight, layer1.bias = layer2.weight, layer2.bias
    else:
        layer1.weight, layer1.bias = \
            nn.Parameter(
                layer2.weight.clone()), nn.Parameter(layer2.bias.clone())
