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


class PositionalEncoding(nn.Module):
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
    """

    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        # Expecting x of 3 dimensions (batch_size, seq_len, embedding_dim)
        # Note: Embedding Dim must be even.
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

        # Now, add to x unsqueezing along batch dimension for broadcast sum
        x = x + PE.unsqueeze(0)  # (batch_size, seq_len, emb_dim) + (1, seq_len, emb_dim)

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
                                        kernel_size=kernel_size, padding=(kernel_size - 1) / 2, groups=in_channels)
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
        self.key_matrix = nn.Linear(embedding_dim, embedding_dim)
        self.query_matrix = nn.Linear(embedding_dim, embedding_dim)
        self.value_matrix = nn.Linear(embedding_dim, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=.1,
                                               batch_first=True)

    def forward(self, x):
        # Expecting x shape (batch_size, seq_len, embedding_dim)
        k = self.key_matrix(x)
        q = self.query_matrix(x)
        v = self.value_matrix(x)
        # Lower triangular mask of ones for value sake.
        mask = torch.tril(torch.ones((x.shape[1], x.shape[2])))
        # We don't need attention weights, just output x (see PyTorch documentation here:
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
        x = self.attention(query=q, key=k, value=v, need_weights=False, attn_mask=mask)
        return x


class FeedForwardNet(nn.Module):
    """
    Implements a simple two-layer MLP since this is the described feed forward net
    in lecture 09 of CS224n (on transformers). Thus, while the authors of the QANet
    do not explicitly describe the FeedForwardNet, this seems most appropriate.
    """

    def __init__(self, embedding_dim, hidden_size):
        super(FeedForwardNet).__init__()
        # Include bias in both linear terms
        self.w1 = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(in_features=hidden_size, out_features=embedding_dim)

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

    def __init__(self, num_conv, input_emb_size, output_emb_size, kernel_size, num_heads):
        super(EncoderBlock, self).__init__()
        assert num_conv > 0, 'There must be at least 1 convolution layer in an Encoder Block'
        self.position_encoder = PositionalEncoding()
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
        self.feed_forward_net = FeedForwardNet(embedding_dim=output_emb_size, hidden_size=output_emb_size)

    def forward(self, x):
        # First, add the PositionalEncoding
        x = x + self.position_encoder(x)

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
                 kernel_size, num_attn_heads):
        super(EncoderStack, self).__init__()
        assert num_blocks > 0, 'There must be at least 1 block i the Embedding Encoder Stack'
        self.encoder_blocks = nn.ModuleList([])
        self.encoder_blocks.append(EncoderBlock(num_conv=num_conv_layers,
                                                input_emb_size=input_emb_size,
                                                output_emb_size=output_emb_size,
                                                kernel_size=kernel_size,
                                                num_heads=num_attn_heads))
        for i in range(num_blocks - 1):
            self.encoder_blocks.append(EncoderBlock(num_conv=num_conv_layers,
                                                    input_emb_size=output_emb_size,
                                                    output_emb_size=output_emb_size,
                                                    kernel_size=kernel_size,
                                                    num_heads=num_attn_heads))

    def forward(self, x):
        # x of shape (batch_size, seq_len, input_emb_size)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        # x of shape (batch_size, seq_len, output_emb_size)
        return x


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
        self.w1 = nn.Linear(in_features=2 * input_size, out_features=1)
        self.w2 = nn.Linear(in_features=2 * input_size, out_features=1)

    def forward(self, m0, m1, m2, mask):
        # M0, M1, M2 will be of shapes (batch_size, seq_len, input_size)
        x1 = torch.cat((m0, m1), dim=2)
        x2 = torch.cat((m0, m2), dim=2)

        x1 = self.w1(x1) # (batch_size, seq_len, 1)
        x2 = self.w2(x2) # (batch_size, seq_len, 1)

        log_p1 = masked_softmax(x1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(x2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

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
