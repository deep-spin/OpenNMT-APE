from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig
from .transformer import TransformerDecoder
import torch.nn as nn
import torch
import numpy as np
import onmt

MAX_SIZE = 512


class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, bert_embeddings, token_type='A'):
        super(MyBertEmbeddings, self).__init__()
        self.word_lut = bert_embeddings.word_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings

        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

        self.token_type = token_type

    def forward(self, input_ids, token_type_ids=None, step=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            if self.token_type is 'A':
                token_type_ids = torch.zeros_like(input_ids)
            else:  # 'B'
                token_type_ids = torch.ones_like(input_ids)
        if step is not None:
            position_ids.fill_(step)

        words_embeddings = self.word_lut(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, bert_layer, init_context=False):
        super(BERTDecoderLayer, self).__init__()
        num_heads = \
            bert_layer.attention.self.num_attention_heads

        hidden_size = \
            bert_layer.attention.self.query.weight.size(0)

        self.init_context = init_context
        self.dropout = bert_layer.attention.self.dropout.p

        # Create self-attention layer
        self.self_attn = onmt.modules.MultiHeadedAttention(
                num_heads, hidden_size, dropout=self.dropout)
        self.self_attn_drop = \
            bert_layer.attention.output.dropout
        self.self_attn_norm = \
            bert_layer.attention.output.LayerNorm

        # Initilaize self-attention layers with bert weights
        self.self_attn.linear_keys = bert_layer.attention.self.key
        self.self_attn.linear_values = bert_layer.attention.self.value
        self.self_attn.linear_query = bert_layer.attention.self.query
        self.self_attn.final_linear = bert_layer.attention.output.dense

        # Create context-attention layer
        self.context_attn = onmt.modules.MultiHeadedAttention(
                num_heads, hidden_size, dropout=self.dropout)
        self.context_attn_drop = \
            bert_layer.attention.output.dropout
        self.context_attn_norm = \
            bert_layer.attention.output.LayerNorm

        if init_context:
            # Initilaize context-attention layers with bert weights
            self.context_attn.linear_keys = bert_layer.attention.self.key
            self.context_attn.linear_values = bert_layer.attention.self.value
            self.context_attn.linear_query = bert_layer.attention.self.query
            self.context_attn.final_linear = bert_layer.attention.output.dense

        self.intermediate = bert_layer.intermediate
        self.output = bert_layer.output

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in BERTDecoderLayer, so
        # it gets BERTDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`

        """
        dec_mask = None
        if step is None:
            dec_mask = torch.gt(tgt_pad_mask +
                                self.mask[:, :tgt_pad_mask.size(-1),
                                          :tgt_pad_mask.size(-1)], 0)

        query, attn = self.self_attn(inputs, inputs, inputs,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query_norm = self.self_attn_norm(self.self_attn_drop(query) + inputs)

        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")

        mid_norm = self.context_attn_norm(
            self.context_attn_drop(mid) + query_norm)
        intermediate_output = self.intermediate(mid_norm)
        output = self.output(intermediate_output, mid_norm)

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class BERTDecoder(TransformerDecoder):
    """
    """
    def __init__(self, num_layers, copy_attn,
                 self_attn_type, vocab_size, pad_idx, init_context=False,
                 token_type='A'):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'bert'
        self.num_layers = num_layers
        self.self_attn_type = self_attn_type
        self.pad_idx = pad_idx
        self.token_type = token_type

        # Decoder State
        self.state = {}

        self._copy = copy_attn

        self.config = BertConfig(vocab_size)
        bert = BertModel(self.config)

        self.embeddings = MyBertEmbeddings(bert.embeddings, token_type)

        self.transformer_layers = nn.ModuleList(
            [BERTDecoderLayer(bert_layer, init_context)
             for bert_layer in bert.encoder.layer])

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.copy_attn,
            opt.self_attn_type,
            embeddings.word_lut.weight.size(0),
            embeddings.word_padding_idx,
            opt.bert_decoder_init_context,
            opt.bert_decoder_token_type)

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        if step == 0:
            self._init_cache(memory_bank)

        src = self.state["src"]
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_words, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        # [B, 1, T_src]
        src_pad_mask = src_words.data.eq(self.pad_idx).unsqueeze(1)
        # [B, 1, T_tgt]
        tgt_pad_mask = tgt_words.data.eq(self.pad_idx).unsqueeze(1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step)

        # Process the result and update the attentions.
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        return dec_outs, attns

    def initialize_bert(self, bert_type):

        bert = BertModel.from_pretrained(bert_type)

        self.embeddings = MyBertEmbeddings(bert.embeddings, self.token_type)

        init_context = self.transformer_layers[0].init_context

        self.transformer_layers = nn.ModuleList(
            [BERTDecoderLayer(bert_layer, init_context)
             for bert_layer in bert.encoder.layer])

        if not init_context:
            for transformer_layer in self.transformer_layers:
                transformer_layer.context_attn.apply(bert.init_bert_weights)
