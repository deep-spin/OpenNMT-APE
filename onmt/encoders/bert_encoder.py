from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig

from onmt.encoders.encoder import EncoderBase


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
        self.bert = BertModel(self.config)
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

        # 0 is padding index in bert models
        mask = src.ne(self.pad_idx)

        encoded_layers, pooled_output = \
            self.bert(src, token_type_ids=segments_ids,
                      attention_mask=mask,
                      output_all_encoded_layers=False)

        return pooled_output.unsqueeze(0),\
            encoded_layers.transpose(0, 1), lengths

    def initialize_bert(self, bert_type):
        self.bert = BertModel.from_pretrained(bert_type)
