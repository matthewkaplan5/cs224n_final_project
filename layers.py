"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu) with modifications by
    Matthew Kaplan, Jane Boettcher, and Jacklyn Luu for
    CS224N Final Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

"""Get appropriate device for cuda support for buffers"""
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# ------------------------------------------------------------------------------------------------------- #
# R-Net Related Layers
class RNetEmbeddings(nn.Module):
    """
    Runs a bi-directional recurrent neural network (RNN) over the character
    embeddings as the input. Concatenates the resulting RNN
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

    Note, that according to the paper it seems the same model is used for both
    the context and the question.
    """
    def __init__(self, char_vectors, word_vectors, hidden_size,
                 num_layers, drop_prob):
        super(RNetEmbeddings, self).__init__()
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.char_rnn = nn.GRU(input_size=char_vectors.shape[1],
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=drop_prob,
                               bidirectional=True)

    def forward(self, char, word):

        word_emb = self.word_embed(word) # word_emb is of shape (N, seq_len, word_emb_dim)
        char_emb = self.char_embed(char) # char_emb is of shape (N, seq_len, word_len, char_emb_dim)

        N = char_emb.shape[0]
        seq_len = char_emb.shape[1]

        # Now reshape to (N * seq_len, word_len, emb_dim)
        char_emb = char_emb.view(char_emb.shape[0] * char_emb.shape[1], char_emb.shape[2], char_emb.shape[3])
        _, char_emb = self.char_rnn(char_emb) # Hidden Out: (2 * num_layers, N * seq_len, hidden_size)

        char_emb = char_emb.transpose(0, 1) # (N * seq_len, 2 * num_layers, hidden_size)
        # Reshape to (N * seq_len, 2 * num_layers * hidden_size)
        char_emb = char_emb.reshape(char_emb.shape[0], char_emb.shape[1] * char_emb.shape[2])
        # Reshape to (N, seq_len, 2 * num_layers * hidden_size)
        char_emb = char_emb.view(N, seq_len, -1)

        # Concatenate to word embeddings
        emb = torch.cat((word_emb, char_emb), dim=2)

        return emb

class GRUEncoder(nn.Module):
    """
    General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Used in the Question and Passage Encoder here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(GRUEncoder, self).__init__()

        self.drop_prob = drop_prob

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.gru(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x

class GatedAttentionBasedRNNVec(nn.Module):
    """
    Trying below without adding the hidden rnn state
    """

    def __init__(self, hidden_size, num_layers, drop_prob):
        super(GatedAttentionBasedRNNVec, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 2 * hidden_size because RNN is bidirectional.
        self.question_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.context_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

        self.gate_proj = nn.Linear(4 * hidden_size, 4 * hidden_size, bias=False)

        self.v_t = nn.Linear(2 * hidden_size, 1, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.drop_prob = drop_prob

        self.gru = nn.GRU(input_size=4 * hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)

    def forward(self, q_emb, c_emb):
        # q_emb shape: (batch_size, q_len, 2 * hidden_size)
        # c_emb shape: (batch_size, c_len, 2 * hidden_size)
        c_len = c_emb.shape[1]
        # Final embedding shape of context
        w_passage = self.context_proj(c_emb) # (batch_size, c_len, 2 * hidden_size)
        w_q = self.question_proj(q_emb).unsqueeze(1).repeat(1, c_len, 1, 1) # (batch_size, c_len, q_len, 2 * hidden_size)
        s = w_q + w_passage.unsqueeze(2) # (N, c_len, q_len, x) + (N, c_len, 1, x)
        s = self.tanh(s)
        s = self.v_t(s).squeeze(dim=3) # (batch_size, c_len, q_len)
        s = F.softmax(s, dim=2)
        # (batch_size, c_len, q_len) x (batch_size, q_len, 2 * hidden_size) --> (batch_size, c_len, 2 * hidden_size)
        s = torch.bmm(s, q_emb)
        # Rocktaschel et al 2015 we send s through the gru at this point, but
        # Wang & Jiang 2016 we take passage and add passage as additional input with gate
        s_concat = torch.cat((c_emb, s), dim=2) # (batch_size, c_len, 4 * hidden_size)
        s = self.gate_proj(s_concat) # (W_g[u_t^P, c_t])
        s = self.sigmoid(s)
        s = torch.mul(s, s_concat) # (batch_size, c_len, 4 * hidden_size) for both element-wise
        s, _ = self.gru(s) # (batch_size, c_len, 2 * hidden_size)

        # No dropout on last layer
        s = F.dropout(s, self.drop_prob, self.training)

        return s

class GatedAttentionBasedRNN(nn.Module):
    """
    This class implementes the Gated Attention-Based Recurrent Networks section 3.2 of the R-Net
    paper here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

    One of the downsides of this layer is given that it is a function of time, a for loop is necessary.
    Otherwise, all elements within the for loop are vectorized.

    Will return v_t^P given vectors u_t^Q and u_t^P which are outputs of the question and passage encoder
    respectively.
    """

    def __init__(self, hidden_size, num_layers, drop_prob):
        super(GatedAttentionBasedRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 2 * hidden_size because RNN is bidirectional.
        self.question_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.context_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.attn_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.gate_proj = nn.Linear(4 * hidden_size, 4 * hidden_size, bias=False)
        self.v_t = nn.Linear(2 * hidden_size, 1, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.drop_prob = drop_prob

        self.gru = nn.GRU(input_size=4 * hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)

    def forward(self, q_emb, c_emb):
        # q_emb shape: (batch_size, q_len, 2 * hidden_size)
        # c_emb shape: (batch_size, c_len, 2 * hidden_size)
        c_len = c_emb.shape[1]
        q_len = q_emb.shape[1]
        # Final embedding shape of context
        emb = torch.zeros_like(c_emb) # (batch_size, c_len, 2 * hidden_size)
        # Zeros like c_emb but with different shape
        h = torch.zeros((2 * self.num_layers, c_emb.shape[0], self.hidden_size),
                        dtype=c_emb.dtype, layout=c_emb.layout, device=c_emb.device)
        w_passage = self.context_proj(c_emb) # (batch_size, c_len, 2 * hidden_size)
        w_q = self.question_proj(q_emb).unsqueeze(1).repeat(1, c_len, 1, 1) # (batch_size, c_len, q_len, 2 * hidden_size)
        w_passage = w_q + w_passage.unsqueeze(2) # (N, c_len, q_len, x) + (N, c_len, 1, x)
        att = torch.zeros_like(c_emb[:, 0, :]) # (batch_size, 2 * hidden_size)
        for step in range(c_len):
            passage = c_emb[:, step, :] # (batch_size, 2 * hidden_size)
            s = self.attn_proj(att).unsqueeze(1) # (batch_size, 1, 2 * hidden_size)
            s = w_passage[:, step, :, :] + s # (batch_size, q_len, 2 * hidden_size)
            s = self.tanh(s)
            s = self.v_t(s).squeeze(dim=2) # (batch_size, q_len)
            s = F.softmax(s, dim=1)
            s = s.unsqueeze(1) # (batch_size, 1, q_len)
            # (batch_size, 1, q_len) x (batch_size, q_len, 2 * hidden_size) --> (batch_size, 1, 2 * hidden_size)
            s = torch.bmm(s, q_emb)
            # Rocktaschel et al 2015 we send s through the gru at this point, but
            # Wang & Jiang 2016 we take passage and add passage as additional input with gate
            s_concat = torch.cat((passage.unsqueeze(1), s), dim=2) # (batch_size, 1, 4 * hidden_size)
            s = self.gate_proj(s_concat) # (W_g[u_t^P, c_t])
            s = self.sigmoid(s)
            s = torch.mul(s, s_concat) # (batch_size, 1, 4 * hidden_size) for both element-wise
            att, h = self.gru(s, h) # (batch_size, 1, 2 * hidden_size)
            att = att.squeeze(dim=1) # (batch_size, 2 * hidden_size)
            emb[:, step, :] = att

        # No dropout on last layer
        emb = F.dropout(emb, self.drop_prob, self.training)

        return emb

class SelfMatchingAttention(nn.Module):
    """
    This implements the Self-Matching Attention layer of R-Net as described at this paper:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

    Since this does not use output of GRU as part of attention computation we can fully vectorize.
    """
    def __init__(self, hidden_size, num_layers, drop_prob):
        super(SelfMatchingAttention, self).__init__()
        self.passage_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.time_proj = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.gate_proj = nn.Linear(4 * hidden_size, 4 * hidden_size, bias=False)
        self.v_t = nn.Linear(2 * hidden_size, 1, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.gru = nn.GRU(input_size=4 * hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)

    def forward(self, emb, lengths):
        # emb is of shape (N, c_len, 2 * hidden_size)
        c_len = emb.shape[1]
        # Reshape to (N, c_len, c_len, 2 * hidden_size) where every array along dimension 1 is the same
        x = emb.unsqueeze(1).repeat(1, c_len, 1, 1)
        x = self.passage_proj(x)
        emb_t_proj = self.time_proj(emb).unsqueeze(2) # (N, c_len, 1, 2 * hidden_size)
        # Broadcast sum along dim 2
        x = x + emb_t_proj # (N, c_len, c_len, 2 * hidden_size)
        x = self.tanh(x)
        x = self.v_t(x).squeeze(dim=3) # (N, c_len, c_len)
        x = F.softmax(x, dim=2)
        # (N, c_len, c_len) x (N, c_len, 2 * hidden_size) --> (N, c_len, 2 * hidden_size)
        x = torch.bmm(x, emb) # (N, c_len, 2 * hidden_size)
        x_concat = torch.cat((emb, x), dim=2) # (N, c_len, 4 * hidden_size)
        x = self.gate_proj(x_concat)
        x = self.sigmoid(x)
        x = torch.mul(x, x_concat)

        x, _ = self.gru(x) # (N, c_len, 2 * hidden_size)

        return x

class RNetOutput(nn.Module):
    """
    Computes the output layer for the R-Net model as described here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
    """
    def __init__(self, batch_size, hidden_size, num_layers, drop_prob):
        super(RNetOutput, self).__init__()
        self.v_r_q = nn.Parameter(torch.zeros((batch_size, 1, 2 * hidden_size)))
        torch.nn.init.xavier_uniform_(self.v_r_q)

        self.w_u = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.w_v = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.w_h_p = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.w_h_a = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.v_t = nn.Linear(2 * hidden_size, 1, bias=False)

        self.tanh = nn.Tanh()

        self.gru = nn.GRU(input_size=2 * hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=drop_prob,
                          bidirectional=True)

    def forward(self, h_emb, q_emb, mask):
        # h_emb (batch_size, c_len, 2 * hidden_size)
        # q_emb (batch_size, q_len, 2 * hidden_size)
        x = self.w_u(q_emb) + self.w_v(self.v_r_q) # (batch_size, q_len, 2 * hidden_size)
        x = self.tanh(x)
        x = self.v_t(x).squeeze(2) # (batch_size, q_len)
        x = F.softmax(x, dim=1)
        x = x.unsqueeze(1) # (batch_size, 1, q_len)
        # (batch_size, 1, q_len) x (batch_size, q_len, 2 * hidden_size) --> (batch_size, 1, 2 * hidden_size)
        x = torch.bmm(x, q_emb)

        x = self.w_h_p(h_emb) + self.w_h_a(x) # (batch_size, c_len, 2 * hidden_size)
        x = self.tanh(x)
        x = self.v_t(x).squeeze(2) # (batch_size, c_len)
        p1 = masked_softmax(x, mask, log_softmax=True)

        # (batch_size, 1, c_len) x (batch_size, c_len, 2 * hidden_size) --> (batch_size, 1, 2 * hidden_size)
        x = torch.bmm(x.unsqueeze(1), h_emb)
        h_1, _ = self.gru(x) # (batch_size, 1, 2 * hidden_size)
        x = self.w_h_p(h_emb) + self.w_h_a(h_1) # (batch_size, c_len, 2 * hidden_size)
        x = self.tanh(x)
        x = self.v_t(x).squeeze(2) # (batch_size, c_len)
        p2 = masked_softmax(x, mask, log_softmax=True)

        return p1, p2

# ------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------- #
# QANet Related Layers
"""
Adds a PositionalEncoding to an input x of shape (batch_size, seq_len, embedding_dim)
as outlined in Attention is All You Need: https://arxiv.org/pdf/1706.03762.pdf

Namely, constructs a Positional Encoder Matrix of shape (seq_len, embedding_dim) where
PE_{ij} = sin(i/10000^(2j / embedding_dim)) for even j and
PE_{ij} = cos(i/10000^(2(j - 1) / embedding_dim)) for odd j

Implementation note:
For vectorization purposes, we take a vector of i: 0 --> seq_len - 1
Then we multiply it to exp(log(i/10000^(2(j - 1) / embedding_dim))) as this allows
us to take away exponent term and use outer product.

To demonstrate, we leverage the fact that:
i * ((1 / 10000)^(2j / embedding_dim))
= i * exp(log((1 / 10000)^(2j / embedding_dim)))
= i * exp(log(1 / 10000) * (2j / embedding_dim))

Leverages broadcast summation to add to x.

We need to build the largest possible position encoder in __init__ and shrink it in the forward
function to fit x, otherwise it will not work on cuda to build the positional encoder in the
forward function.
"""
def position_encoder(x):
    seq_len = x.shape[1]
    emb_dim = x.shape[2]

    # First get positions from 1 to sequence length
    pos = torch.tensor(range(seq_len))

    # Then, get the next term without position multiplied to it. We only want even
    # columns. Need to use numpy as torch.log only accepts tensors.
    encoder = torch.exp(np.log(1 / 10000) * (2 * torch.tensor(range(0, emb_dim, 2))) / emb_dim)

    PE = torch.zeros((seq_len, emb_dim))
    # Now, outer product the position and encoder (of even j) into the odd
    # and even columns of PE respectively. Then, as described, for
    # even columns take sin, odd columns take cos.
    PE[:, range(0, emb_dim, 2)] = torch.sin(torch.outer(pos, encoder))
    PE[:, range(1, emb_dim, 2)] = torch.cos(torch.outer(pos, encoder))

    PE = PE.to(device)

    return PE

class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Matthew, Jane, Jacklyn modification: Integrate character level embeddings
    per the QANet paper (char_embedding size of 200 after convolutions).
    Uses a CNN net over the character embeddings, before appending to the
    word embeddings (which then goes through projection layer

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, char_vectors, word_vectors,
                 char_emb_size, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.char_conv = CharCNN(in_channels=char_vectors.size(1), output_size=char_emb_size)
        self.proj = nn.Linear(word_vectors.size(1) + char_emb_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, char, word):
        char_emb = self.char_embed(char)  # (batch_size, seq_len, max_word_len, char_emb_size)
        word_emb = self.word_embed(word)  # (batch_size, seq_len, word_embed_size)

        char_emb = self.char_conv(char_emb)  # (batch_size, seq_len, word_emb_size)

        char_emb = F.dropout(char_emb, self.drop_prob, self.training)
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)

        # Concatenate here:
        emb = torch.cat((word_emb, char_emb), dim=2)  # (batch_size, seq_len, word_emb + char_emb)

        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)

        return emb


class CharCNN(nn.Module):
    """
    Run Character Embeddings through a Convolutional Neural Network before
    concatenating to word embeddings as described by Kim (2014). Appended
    to word embeddings eventually as described in QANet paper (linked below).

    Inspired by Kim et al. slides to accompany paper here, but does not use
    tanh nonlinearity as described in these slides:
    https://people.csail.mit.edu/dsontag/papers/kim_etal_AAAI16_slides.pdf

    Model embeddings meant to fit description and dimensions as described
    in the QANet paper here:
    https://arxiv.org/pdf/1804.09541.pdf

    A simple conv1d layer over characters in given word, followed by a
    maxpool layer to remove dimension (We just use torch.max).
    """

    def __init__(self, in_channels, output_size, conv_filter_size=3):
        super(CharCNN, self).__init__()
        # The out_channels is the number of filters (filter have width embedding length, height conv_filter_size).
        self.conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=output_size, kernel_size=conv_filter_size)

    def forward(self, x):
        # Expecting 4 dimensional char dim.
        b = x.shape[0]
        seq_len = x.shape[1]
        # Transpose as middle index must be embedding size (as this is width of conv filter)
        x = x.view(b * seq_len, x.shape[2], x.shape[3]).transpose(1, 2)  # (batch_size * seq_len, char_emb_size, max_word_len)

        x = self.conv_layer(x)  # (batch_size * seq_len, out_channels, conv_out)

        # Reshape x back into 4 dims
        x = x.view(b, seq_len, x.shape[1], -1)  # (batch_size, seq_len, out_channels, conv_out)

        # Take max over rows of (out_channels, conv_out) matrix as described in QANet paper.
        # QANet paper describes characters concatenated row-wise to make word (what we have
        # here when considering matrix of last 2 dimensions after transposing). Thus, we take
        # max over 3rd dimension (rows of matrix made up by final 2 dimensions).

        # We could have used maxpool but max is better as it will always remove last dimension
        # regardless of kernel size (we do not have to manually adjust maxpool filter size).
        x, _ = torch.max(x, dim=3)  # (batch_size, seq_len, out_channels)

        return x


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class DepthwiseSeparableConvolutions(nn.Module):
    """
    Implements Depthwise Separable Convolutions as outlined by Chollet (2017)
    at https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
    and further explained at this nice article here:
    https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

    Note that we are using this as part of implementing the model defined in the
    QANet Paper: https://arxiv.org/pdf/1804.09541.pdf where the authors use a
    depthwise kernel size of 7.

    A Couple Implementation Notes

    1) As the authors of QANet state, they map the character embeddings from hidden_size to
    d = 128, which is the number of filters. That means the channel dimension of x must be
    the character embeddings, while the Length dimension is the sequence length (which we must maintain)

    2) Given we use this for multiple layers, we want the output_dimension to equal our input dim.
    From PyTorch Conv1d Documentation, https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html we see:
    output_dim = ((input_dim + (2 * padding) - 1 x (kernel_size - 1) - 1) / 1) + 1
    = input_dim + (2 * padding) - kernel_size + 1.

    Thus, for input_dim to equal output_dim, we see that
    (2 * padding) - kernel_size + 1 = 0
    padding = (kernel_size - 1) / 2.

    Thus, we need kernel_size to be an odd number, with padding of (kernel_size - 1) / 2 to maintain
    the dimensions.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConvolutions, self).__init__()
        # Kernel Size must be odd to retain dimensions. If even, increment by 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # With help from this article: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
        # See the "Part 1 - Depthwise Convolution"
        # and "Part 2 - Pointwise Convolution" sections.

        # Need in_channel conv filters of 1 channel x width of 7 (the groups argument allows us to alter channels).
        # Explanation for padding in comments above. Implicitly concatenated on channel axis.
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=kernel_size, padding=int((kernel_size - 1) / 2), groups=in_channels)
        # Further with help from paper, we now have (in_channels x 1) filters. this is
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # x is shape (batch_size, seq_len, embedding_dim)

        # Transpose x to (batch_size, embedding_dim, seq_len) as QANet paper describes
        # we make embedding_dim the channel dimension and seq_len the length dimension.
        x = x.transpose(1, 2)

        # Apply the convolutions as described
        # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
        x = self.depthwise_conv(x)  # (batch_size, embedding_dim, seq_len)
        x = self.pointwise_conv(x)  # (batch_size, out_channels, seq_len)

        # Transpose x back to # (batch_size, seq_len, out_channels)
        x = x.transpose(1, 2)

        return x


class QANetSelfAttention(nn.Module):
    """
    This implements a simple self-attention module used in the original
    paper for Transformers "Attention is All You Need":
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_dim, num_heads):
        super(QANetSelfAttention, self).__init__()
        self.key_matrix = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query_matrix = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_matrix = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=.1,
                                               batch_first=True)

    def forward(self, x):
        # Expecting x shape (batch_size, seq_len, embedding_dim)
        k = self.key_matrix(x)
        q = self.query_matrix(x)
        v = self.value_matrix(x)
        # Lower triangular mask of ones for value sake.
        mask = torch.tril(torch.ones((x.shape[1], x.shape[1]))).to(device)
        # We don't need attention weights, just output x (see PyTorch documentation here:
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
        x = self.attention(query=q, key=k, value=v, need_weights=False, attn_mask=mask)
        # Returns a tuple (x, None)
        return x[0]

class FeedForwardNet(nn.Module):
    """
    Implements a simple two-layer MLP since this is the described feed forward net
    in lecture 09 of CS224n (on transformers). Thus, while the authors of the QANet
    do not explicitly describe the FeedForwardNet, this seems most appropriate.
    """

    def __init__(self, embedding_dim, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        # Include bias in both linear terms
        self.w1 = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # (w_1 * x + b_1)
        x = self.w1(x)
        # ReLU(w_1 * x + b_1)
        x = self.relu(x)
        # W_2 * ReLU(w_1 * x + b_1) + b_2
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    """
    A single Encoder Block as shown on the right side of figure 1 of the
    QANet paper: https://arxiv.org/pdf/1804.09541.pdf

    Basic Architecture:
    Position Encoding + (LayerNorm + DepthwiseSeparable Conv) x num_conv + (LayerNorm + SelfAttention)
    + (Layernorm + FeedForwardNet)

    Note, there is full identity path from input to output of each block, meaning for conv/selfattention/ffn
    functions f, we perform f(layernorm(x)) + x
    """

    def __init__(self, num_conv, input_emb_size, output_emb_size, kernel_size, num_heads,
                 ffn_hidden_size):
        super(EncoderBlock, self).__init__()
        assert num_conv > 0, 'There must be at least 1 convolution layer in an Encoder Block'
        # self.position_encoder = PositionalEncoding(emb_dim=input_emb_size)
        self.conv_layers = nn.ModuleList([])
        # First conv layer input --> output.
        self.conv_layers.append(DepthwiseSeparableConvolutions(in_channels=input_emb_size,
                                                               out_channels=output_emb_size,
                                                               kernel_size=kernel_size))
        for i in range(num_conv - 1):
            # The rest of the layers are output --> output
            self.conv_layers.append(DepthwiseSeparableConvolutions(in_channels=output_emb_size,
                                                                   out_channels=output_emb_size,
                                                                   kernel_size=kernel_size))
        self.self_attention = QANetSelfAttention(embedding_dim=output_emb_size,
                                                 num_heads=num_heads)
        self.feed_forward_net = FeedForwardNet(embedding_dim=output_emb_size, hidden_size=ffn_hidden_size,
                                               output_size=output_emb_size)

    def forward(self, x):
        # First, add the PositionalEncoding
        x += position_encoder(x)

        # First, repeat conv(layernorm(x)) + x for all conv layers
        for conv in self.conv_layers:
            out = x.clone()
            x = F.layer_norm(x, x.shape[1:])
            x = conv(x)
            x = x + out

        # Now, self_attention(layernorm(x)) + x
        out = x.clone()
        x = F.layer_norm(x, x.shape[1:])
        x = self.self_attention(x)
        x = x + out

        # Now, feed_forward_net(layernorm(x)) + x
        out = x.clone()
        x = F.layer_norm(x, x.shape[1:])
        x = self.feed_forward_net(x)
        x = x + out

        return x


class EncoderStack(nn.Module):
    """
    This stacks Encoder Blocks for the embeddings and model, illustrated by the
    "Stacked Embedding Encoder Blocks" and "Stacked Model Encoder Blocks"
    section of figure 4 of the QANet paper: https://arxiv.org/pdf/1804.09541.pdf
    """

    def __init__(self, num_blocks, num_conv_layers, input_emb_size, output_emb_size,
                 kernel_size, num_attn_heads, ffn_hidden_size):
        super(EncoderStack, self).__init__()
        assert num_blocks > 0, 'There must be at least 1 block i the Embedding Encoder Stack'
        self.encoder_blocks = nn.ModuleList([])
        self.encoder_blocks.append(EncoderBlock(num_conv=num_conv_layers,
                                                input_emb_size=input_emb_size,
                                                output_emb_size=output_emb_size,
                                                kernel_size=kernel_size,
                                                num_heads=num_attn_heads,
                                                ffn_hidden_size=ffn_hidden_size))
        for i in range(num_blocks - 1):
            self.encoder_blocks.append(EncoderBlock(num_conv=num_conv_layers,
                                                    input_emb_size=output_emb_size,
                                                    output_emb_size=output_emb_size,
                                                    kernel_size=kernel_size,
                                                    num_heads=num_attn_heads,
                                                    ffn_hidden_size=ffn_hidden_size))

    def forward(self, x):
        # x of shape (batch_size, seq_len, input_emb_size)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        # x of shape (batch_size, seq_len, output_emb_size)
        return x

class QANetOutput(nn.Module):
    """
    This is the output layer used by the authors of QANet here:
    https://arxiv.org/pdf/1804.09541.pdf

    We use a linear layer over concatenated inputs before applying
    a softmax to generate log probabilities (for the nll loss).
    """
    def __init__(self, input_size):
        super(QANetOutput, self).__init__()
        # From paper, w1 and w2 matrices with concatenated inputs that
        # we want to send to 1 (so we can squeeze into 2 dimensions)
        self.w1 = nn.Linear(in_features=2 * input_size, out_features=1, bias=False)
        self.w2 = nn.Linear(in_features=2 * input_size, out_features=1, bias=False)

    def forward(self, m0, m1, m2, mask):
        # M0, M1, M2 will be of shapes (batch_size, seq_len, input_size)
        x1 = torch.cat((m0, m1), dim=2)
        x2 = torch.cat((m0, m2), dim=2)

        x1 = self.w1(x1) # (batch_size, seq_len, 1)
        x2 = self.w2(x2) # (batch_size, seq_len, 1)

        p1 = masked_softmax(x1.squeeze(), mask, log_softmax=True)
        p2 = masked_softmax(x2.squeeze(), mask, log_softmax=True)

        return p1, p2

# ------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------- #
# BiDAF Related Layers (BiDAF Attention used by QANet)
class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    This is also used by QANet here: https://arxiv.org/abs/1804.09541
    Except QANet returns a and b separately, and concatenates a, b, and c
    as done in this function later, this returns the same concatenation now.

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

# ------------------------------------------------------------------------------------------------------- #