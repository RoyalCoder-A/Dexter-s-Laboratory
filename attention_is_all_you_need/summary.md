# Attentions is all you need summary

## Abstract
On abstract, this paper is saying it is a encoder decoder architecture, named `Transformer` which is much faster
and more accurate than normal RNN models. (They test it on an german to english translation task)

## Introduction
So here, they're saying that RNNs have an issue, which is each step `h_t` is dependent to `h_t - 1` and input of `t`
which is problematic for parallel calculations and also if we have a long sequence, on end of the sequence, if we want
to remember the context of the beginning of the sequence, we should have lot of parameters on our RNN layers. So on 
`Transformer` we can have a global attention between each input and output, so no need for huge RNN layers.


## Architecture
![architecture image](./architecture.png)

### Encoder:
As they mention, encoder is N stack of layers (which N is 6) and it includes
two sub layers (left side of the image) first is the multi head attention and second
is a simple fully connected positional wise feed forward. Also, both of them are followed
by a residual connection and a layer normalization. They mention to facilitate these residual
connections, all sub layers (including embeddings) will return dimension of 512

### Decoder:
Decoder is also an N layer stack (which N is 6). It has another sub layer in addition to 
the encoder layer, which is a masked multi head attention to the output of encoder. The reason
for masking, as they mentioned, is to prevent model to look forward positions for each i.
It's hard to explain, but it's like when we are predicting market candles, we shouldn't look in the future,
that's same here. I can understand the big picture, but how is that going to happen, I should first know about
input and outputs of each layer to be able to fully understand this.


### Attention:
So attention, as the paper is describing, is a mapping a query and a set of key-values
to an output. So what does that mean? It means you have a query vector, a key vector, and a
value vector. The output is weighted sum of the values, and what are the weights? It will compute by a
function of the query with its corresponding keys.