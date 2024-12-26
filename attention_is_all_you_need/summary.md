# Attentions is all you need summary

## Abstract
On abstract, this paper is saying it is a encoder decoder architecture, named `Transformer` which is much faster
and more accurate than normal RNN models. (They test it on an german to english translation task)

## Introduction
So here, they're saying that RNNs have an issue, which is each step `h_t` is dependent to `h_t - 1` and input of `t`
which is problematic for parallel calculations and also if we have a long sequence, on end of the sequence, if we want
to remember the context of the beginning of the sequence, we should have lot of parameters on our RNN layers. So on 
`Transformer` we can have a global attention between each input and output, so no need for huge RNN layers.