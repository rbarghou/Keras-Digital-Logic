# Keras-Digital-Logic
A repository demonstrating the use of RNNs written in Keras to build neural network analogs to simple digital logic circuits.

## Goal
The goal of this repository is to document some techniques for training simple Keras Recurrent Neural Netorks on the behaviors common in Digital Logic Gate circuits like Flip Flops, D-Latches, Edge Detectors, Bit Shift Registers, and Clocks.

## What's the point?
Much work has been done with LSTM networks and GRU networks lately, and these are very powerful RNN designs.  However, they are also more complicated to implement and come with some performance tradeoffs over Simple RNN designs.  I believe it may be possible to recover some of the performance of a Simple RNN while still preserving some of the more advanced functionality of these more complicated gating schemes.

My hypothesis is that by training networks on Digital Logic Gates, and then using these trained networks in more sophisticated initialization schemes for Simple RNNs, training such networks could be much faster while still allowing for the flexibility that Simple RNNs permit.

## What's in this repo:
Nothing yet.

Wish me luck!
