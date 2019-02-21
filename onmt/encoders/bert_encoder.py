from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig
import torch.nn as nn
import torch

from onmt.encoders.encoder import EncoderBase


class MyEncoderBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, bert_embeddings):
        super(MyEncoderBertEmbeddings, self).__init__()
        self.word_lut = bert_embeddings.word_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings

        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # The point of this code block is to reset the position ids
        # for sentence B (token_type_id=1)
        token_type = (token_type_ids > 0)
        position_aux = \
            ((token_type.cumsum(1) == 1) & token_type).max(1)[1].unsqueeze(1)
        position_aux = position_aux * token_type_ids.clone()
        position_ids = position_ids - position_aux

        words_embeddings = self.word_lut(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTEncoder(EncoderBase):
    """
    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, vocab_size, pad_idx):
        super(BERTEncoder, self).__init__()
        self.config = BertConfig(vocab_size)
        bert = BertModel(self.config)
        self.embeddings = \
            MyEncoderBertEmbeddings(bert.embeddings)
        self.encoder = bert.encoder

        self.pad_idx = pad_idx

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings.word_lut.weight.size(0),
            embeddings.word_padding_idx
        )

    def forward(self, src, lengths=None, **kwargs):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # bert receives a tensor of shape [batch_size x src_len]
        segments_ids = src[:, :, 1].t()
        src = src[:, :, 0].t()

        attention_mask = src.ne(self.pad_idx)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular
        # masking of causal attention
        # used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend
        # and 0.0 for masked positions, this operation will create a
        # tensor which is 0.0 for positions we want to attend
        # and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = \
            extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(src, segments_ids)

        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)

        encoded_layers = encoded_layers[-1]

        return embedding_output,\
            encoded_layers.transpose(0, 1), lengths

    def initialize_bert(self, bert_type):

        bert = BertModel.from_pretrained(bert_type)

        self.embeddings = \
            MyEncoderBertEmbeddings(bert.embeddings)

        self.encoder = bert.encoder
