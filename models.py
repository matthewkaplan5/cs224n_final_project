"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    char_emb_size=200,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = cw_mask.sum(-1), qw_mask.sum(-1)

        c_emb = self.emb(cc_idxs, cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qc_idxs, qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       cw_mask, qw_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, cw_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    """
    The QANet Model Paper as originally written by Yu et al. 2018
    https://arxiv.org/pdf/1804.09541.pdf

    Written from scratch following the paper by Matthew Kaplan,
    Jane Boettcher, and Jacklyn Luu for CS224N Default Final Project
    on the provided SQuAD Dataset.
    """
    def __init__(self, char_vectors, word_vectors, hidden_size, drop_prob=0.1):
        super(QANet, self).__init__()

        self.emb = layers.Embedding(char_vectors=char_vectors,
                                    word_vectors=word_vectors,
                                    char_emb_size=200,
                                    hidden_size=500, # Same as QANet Paper
                                    drop_prob=drop_prob)

        self.emb_enc_1 = layers.EncoderStack(num_blocks=1,
                                             num_conv_layers=4,
                                             input_emb_size=500,
                                             output_emb_size=hidden_size,
                                             kernel_size=7,
                                             num_attn_heads=8)

        self.emb_enc_2 = layers.EncoderStack(num_blocks=1,
                                             num_conv_layers=4,
                                             input_emb_size=500,
                                             output_emb_size=hidden_size,
                                             kernel_size=7,
                                             num_attn_heads=8)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.model_enc = layers.EncoderStack(num_blocks=7,
                                             num_conv_layers=2,
                                             input_emb_size=8 * hidden_size,
                                             output_emb_size=hidden_size,
                                             kernel_size=7,
                                             num_attn_heads=8)

        self.out = layers.QANetOutput(input_size=hidden_size)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        cw_mask = torch.zeros_like(cw_idxs) != cw_idxs
        qw_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_emb = self.emb(cc_idxs, cw_idxs)  # (batch_size, c_len, 500)
        q_emb = self.emb(qc_idxs, qw_idxs)  # (batch_size, q_len, 500)

        c_enc = self.emb_enc_1(c_emb) # (batch_size, c_len, hidden_size)
        q_enc = self.emb_enc_2(q_emb) # (batch_size, q_len, hidden_size)

        c_enc = F.dropout(c_enc, p=0.1, training=self.training)
        q_enc = F.dropout(q_enc, p=0.1, training=self.training)

        att = self.att(c_enc, q_enc, cw_mask, qw_mask)  # (batch_size, c_len, 8 * hidden_size)

        att = F.dropout(att, p=0.1, training=self.training)

        m_0 = self.model_enc(att) # (batch_size, c_len, hidden_size)
        m_1 = self.model_enc(m_0.clone()) # (batch_size, c_len, hidden_size)
        m_2 = self.model_enc(m_1.clone()) # (batch_size, c_len, hidden_size)

        out = self.out(m_0, m_1, m_2, cw_mask) # (batch_size, c_len)

        return out
