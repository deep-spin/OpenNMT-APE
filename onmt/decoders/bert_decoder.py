from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig
from .transformer import TransformerDecoder
import torch.nn as nn
import torch
import numpy as np
import onmt
import copy

MAX_SIZE = 512


def clone_or_share_layer(layer1, layer2, share=False):

    if share:
        layer1.weight, layer1.bias = layer2.weight, layer2.bias
    else:
        layer1.weight, layer1.bias = \
            nn.Parameter(
                layer2.weight.clone()), nn.Parameter(layer2.bias.clone())


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

    def __init__(self, bert_layer, init_context=False,
                 context_att_type='concat', double_residual=False):
        super(BERTDecoderLayer, self).__init__()
        num_heads = \
            bert_layer.attention.self.num_attention_heads

        hidden_size = \
            bert_layer.attention.self.query.weight.size(0)

        self.init_context = init_context
        self.context_att_type = context_att_type
        self.double_residual = double_residual
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

        # Create context-attention layer 1
        self.context_attn = onmt.modules.MultiHeadedAttention(
                num_heads, hidden_size, dropout=self.dropout)
        self.context_attn_drop = \
            bert_layer.attention.output.dropout
        self.context_attn_norm = \
            bert_layer.attention.output.LayerNorm

        if init_context:
            # Initilaize context-attention layers with bert weights
            clone_or_share_layer(
                self.context_attn.linear_keys,
                bert_layer.attention.self.key,
                share=False
            )
            clone_or_share_layer(
                self.context_attn.linear_values,
                bert_layer.attention.self.value,
                share=False
            )
            clone_or_share_layer(
                self.context_attn.linear_query,
                bert_layer.attention.self.query,
                share=False
            )
            clone_or_share_layer(
                self.context_attn.final_linear,
                bert_layer.attention.output.dense,
                share=False
            )

        if self.context_att_type != 'concat':
            # Create context-attention layer 2
            self.context_attn_2 = onmt.modules.MultiHeadedAttention(
                    num_heads, hidden_size, dropout=self.dropout)
            self.context_attn_2_drop = \
                bert_layer.attention.output.dropout
            self.context_attn_2_norm = \
                bert_layer.attention.output.LayerNorm

            if init_context:
                # Initilaize context-attention layers with bert weights
                clone_or_share_layer(
                    self.context_attn_2.linear_keys,
                    bert_layer.attention.self.key,
                    share=False
                )
                clone_or_share_layer(
                    self.context_attn_2.linear_values,
                    bert_layer.attention.self.value,
                    share=False
                )
                clone_or_share_layer(
                    self.context_attn_2.linear_query,
                    bert_layer.attention.self.query,
                    share=False
                )
                clone_or_share_layer(
                    self.context_attn_2.final_linear,
                    bert_layer.attention.output.dense,
                    share=False
                )

        nn.ModuleList([copy.deepcopy(self.context_attn) for _ in range(2)])

        self.intermediate = bert_layer.intermediate
        self.output = bert_layer.output

        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in BERTDecoderLayer, so
        # it gets BERTDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None,
                sent_B_memory_bank=None, sent_B_pad_mask=None):
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

        if self.context_att_type == 'AB':
            memory_bank = [memory_bank, sent_B_memory_bank]
            src_pad_mask = [src_pad_mask, sent_B_pad_mask]
        elif self.context_att_type == 'BA':
            memory_bank = [sent_B_memory_bank, memory_bank]
            src_pad_mask = [sent_B_pad_mask, src_pad_mask]
        else:
            memory_bank = [memory_bank]
            src_pad_mask = [src_pad_mask]

        mid, attn = self.context_attn(memory_bank[0], memory_bank[0],
                                      query_norm,
                                      mask=src_pad_mask[0],
                                      layer_cache=layer_cache,
                                      type="context")

        mid_norm = self.context_attn_norm(
            self.context_attn_drop(mid) + query_norm)

        if self.context_att_type != 'concat':
            mid, attn = self.context_attn_2(
                memory_bank[1], memory_bank[1], mid_norm,
                mask=src_pad_mask[1],
                layer_cache=layer_cache,
                type="context")

            residual_connection = query_norm if self.double_residual \
                else mid_norm
            mid_norm = self.context_attn_2_norm(
                self.context_attn_2_drop(mid) + residual_connection)

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
    def __init__(self, copy_attn, vocab_size, pad_idx,
                 init_context=False, context_att_type='concat',
                 double_residual=False, token_type='A'):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'bert'
        self.pad_idx = pad_idx
        self.token_type = token_type
        self.init_context = init_context
        self.context_att_type = context_att_type
        self.double_residual = double_residual

        # Decoder State
        self.state = {}

        self._copy = copy_attn

        self.config = BertConfig(vocab_size)
        bert = BertModel(self.config)

        self.embeddings = MyBertEmbeddings(bert.embeddings, token_type)

        self.transformer_layers = nn.ModuleList(
            [BERTDecoderLayer(bert_layer, init_context,
                              context_att_type, double_residual)
             for bert_layer in bert.encoder.layer])

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.copy_attn,
            embeddings.word_lut.weight.size(0),
            embeddings.word_padding_idx,
            opt.bert_decoder_init_context,
            opt.bert_decoder_context_att_type,
            opt.transformer_decoder_double_residual,
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

        segments_ids = src[:, :, 1].transpose(0, 1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_words, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        # [B, 1, T_src]
        src_pad_mask = src_words.data.eq(self.pad_idx).unsqueeze(1)
        # [B, 1, T_tgt]
        tgt_pad_mask = tgt_words.data.eq(self.pad_idx).unsqueeze(1)

        if self.context_att_type != 'concat':
            sent_A_memory_bank, sent_B_memory_bank, \
                sent_A_pad_mask, sent_B_pad_mask = \
                self.split_sent_A_sent_B(segments_ids, src_memory_bank)

            src_memory_bank = sent_A_memory_bank
            src_pad_mask = sent_A_pad_mask.type_as(src_pad_mask)
        else:
            sent_B_memory_bank, sent_B_pad_mask = None, None

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
                sent_B_memory_bank=sent_B_memory_bank,
                sent_B_pad_mask=sent_B_pad_mask
                )

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

        self.transformer_layers = nn.ModuleList(
            [BERTDecoderLayer(bert_layer, self.init_context,
                              self.context_att_type, self.double_residual)
             for bert_layer in bert.encoder.layer])

        if not self.init_context:
            for transformer_layer in self.transformer_layers:
                transformer_layer.context_attn.apply(bert.init_bert_weights)

    def split_sent_A_sent_B(self, segments_ids, src_memory_bank):

        token_type = (segments_ids > 0)
        sent_A_lengths = ((token_type.cumsum(1) == 1) & token_type).max(1)[1]
        sent_A_memory_bank = ((segments_ids == 0).type_as(
            src_memory_bank
            ).unsqueeze(-1)*src_memory_bank)[:, :sent_A_lengths.max()]

        unrolled_sent_B = ((segments_ids == 1).type_as(
            src_memory_bank).unsqueeze(-1) * src_memory_bank)
        sent_B_lengths = (segments_ids == 1).sum(dim=1)

        extra_padding = \
            segments_ids.size(1) - (sent_A_lengths + sent_B_lengths)

        sent_B_memory_bank = torch.stack([
            torch.roll(
                x_i,
                shifts=int(sent_B_lengths[i] + extra_padding[i]),
                dims=0
                ) for i, x_i in enumerate(
                    torch.unbind(unrolled_sent_B, dim=0))], dim=0)
        sent_B_memory_bank = sent_B_memory_bank[:, :sent_B_lengths.max()]

        sent_A_pad_mask = segments_ids[:, :sent_A_lengths.max()].unsqueeze(1)
        sent_B_pad_mask = \
            torch.stack([torch.roll(
                x_i,
                shifts=int(sent_B_lengths[i] + extra_padding[i]),
                dims=0
                ) for i, x_i in enumerate(
                    torch.unbind(segments_ids, dim=0))], dim=0)
        sent_B_pad_mask = \
            sent_B_pad_mask[:, :sent_B_lengths.max()].eq(0).unsqueeze(1)

        return sent_A_memory_bank, sent_B_memory_bank, \
            sent_A_pad_mask, sent_B_pad_mask
