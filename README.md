# PyTorch Tutorials

This repo contains my implementations and notes on some tutorials for [PyTorch](https://pytorch.org/). 

## Tutorials
1 - [Learning PyTorch](https://github.com/syuoni/PyTorch-Tutorials/tree/master/Learning-PyTorch) follows the [Official Tutorials](https://pytorch.org/tutorials/) in the `Learning PyTorch` collection. 
* [60 Minute Blitz 1](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Learning-PyTorch/%5B1%5D60-Min-Blitz-1-PyTorch-Basics.ipynb) 
* [60 Minute Blitz 2](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Learning-PyTorch/%5B2%5D60-Min-Blitz-2-Neural-Networks.ipynb)
* [Learning PyTorch with Examples](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Learning-PyTorch/%5B3%5DLearning-PyTorch-with-Examples.ipynb)
* [What is torch.nn Really](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Learning-PyTorch/%5B4%5DWhat-is-torch.nn-Really.ipynb)
* [Visualization with TensorBoard](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Learning-PyTorch/%5B5%5DVisualization-with-TensorBoard.ipynb)

2 - [Text](https://github.com/syuoni/PyTorch-Tutorials/tree/master/Text) follows the [Official Tutorials](https://pytorch.org/tutorials/) in the `Text` collection.
* [Intro to PyTorch](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B1%5DIntro-to-PyTorch.ipynb)
* [Word Embeddings](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B2%5DWord-Embeddings.ipynb)
* [LSTM and Other RNNs](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B3%5DLSTM-and-Other-RNNs.ipynb)
* [Classifying Names](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B4%5DClassifying-Names.ipynb): Char-level one-hot embeddings into an RNN, to predict name categories. 
* [Generating Names](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B5%5DGenerating-Names.ipynb): Char-level and category one-hot embeddings into an RNN, to recurrently predict next characters. 
* [Translation with Seq2Seq and Attention](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B6%5DTranslation-with-Seq2Seq-Net-and-Atten.ipynb): Encoder-decoder seq2seq framework with attention. 
* [Classifying Text with TorchText](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B7%5DClassifying-Text-with-TorchText.ipynb): An implementation of `FastText` using `TorchText`. 
* [Translating Language with TorchText](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B8%5DTranslating-Language-with-TorchText.ipynb): Encoder-decoder seq2seq framework with attention using `TorchText`. 
* [BiLSTM CRF](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Text/%5B9%5DBiLSTM-CRF.ipynb): Conditional Random Field (CRF) implementation. 

3 - [Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/tree/master/Sentiment-Analysis) follows the [PyTorch Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis) repo by [Ben Trevett](https://github.com/bentrevett). 
* [Simple Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B1%5DSimple-Sentiment-Analysis.ipynb): Word embeddings into an RNN, to predict a binary sentiment target. 
    * Intro to `Field`, `Example`, `Dataset` and `BucketIterator` of `TorchText`. 
* [Upgraded Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B2%5DUpgraded-Sentiment-Analysis.ipynb): Pre-trained embeddings, bidirectional multi-layer LSTM, dropout and `<pad>` token handling. 
    * Intro to `Vectors` and `Vocab` of `TorchText`. 
* [Faster Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B3%5DFaster-Sentiment-Analysis.ipynb): An implementation of `FastText`. 
* [Convolutional Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B4%5DConvolutional-Sentiment-Analysis.ipynb): Word embeddings into a CNN, to predict a binary sentiment target. 
* [Multi-Class Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B5%5DMulti-Class-Sentiment-Analysis.ipynb): Word embeddings into a CNN, to predict a multi-class sentiment target. 
* [Transformers for Sentiment Analysis](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5B6%5DTransformers-for-Sentiment-Analysis.ipynb): Using pre-trained `BERT` (by [huggingface](https://huggingface.co/transformers/)) as contextual word embeddings, followed by an RNN, to predict a binary sentiment target. 
* [TorchText Notes](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Sentiment-Analysis/%5BA%5DTorchText-Notes.ipynb): Details of `BucketIterator.splits`, etc. 

4 - [Seq2Seq](https://github.com/syuoni/PyTorch-Tutorials/tree/master/Seq2Seq) follows the [PyTorch Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq) repo by [Ben Trevett](https://github.com/bentrevett). 
* [Seq2Seq with Neural Networks](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B1%5DSeq2Seq-with-NN.ipynb): Basic encoder-decoder framework. 
* [Learning Phrase Representations](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B2%5DLearning-Phrase-Representations.ipynb): Relieving information compression by using context vector in every decoding step.  
* [Learning to Align and Translate](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B3%5DLearning-to-Align-and-Translate.ipynb): Further relieving information compression by attention. 
* [Convolutional Seq2Seq Learning](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B5%5DConv-Seq2Seq-Learning.ipynb): Using CNN instead of RNN for encoder and decoder. 
* [Attention Is All You Need](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B6%5DAttention-Is-All-You-Need.ipynb): An implementation of `Transformer`. 
* [Transformer by PyTorch](https://github.com/syuoni/PyTorch-Tutorials/blob/master/Seq2Seq/%5B7%5DTransformer-by-PyTorch.ipynb): Using `torch.nn.Transformer`. 

5 - [PoS Tagging](https://github.com/syuoni/PyTorch-Tutorials/tree/master/PoS-Tagging) follows the [PyTorch PoS Tagging](https://github.com/bentrevett/pytorch-pos-tagging) repo by [Ben Trevett](https://github.com/bentrevett). 
* [BiLSTM for PoS Tagging](https://github.com/syuoni/PyTorch-Tutorials/blob/master/PoS-Tagging/%5B1%5DBiLSTM-for-PoS-Tagging.ipynb): Word embeddings into a BiLSTM, to predict the tag sequence (position by position). 
* [Fine Tuning Pretrained Transformers for PoS Tagging](https://github.com/syuoni/PyTorch-Tutorials/blob/master/PoS-Tagging/%5B2%5DFine-Tuning-Pretrained-Transformers-for-PoS-Tagging.ipynb): Using pre-trained `BERT` (by [huggingface](https://huggingface.co/transformers/)) as contextual word embeddings, to predict the tag sequence. 
* [BiLSTM-CRF for PoS Tagging](https://github.com/syuoni/PyTorch-Tutorials/blob/master/PoS-Tagging/%5B3%5DBiLSTM-CRF-for-PoS-Tagging.ipynb): Word embeddings into a BiLSTM followed by a CRF layer, to predict the tag sequence. 
* [CRF by torchcrf](https://github.com/syuoni/PyTorch-Tutorials/blob/master/PoS-Tagging/%5B4%5DCRF-by-torchcrf.ipynb): Using [pytorch-crf](https://pytorch-crf.readthedocs.io/en/stable/) package for CRF. 
* [BiLSTM for NER](https://github.com/syuoni/PyTorch-Tutorials/blob/master/PoS-Tagging/%5B5%5DBiLSTM-for-NER.ipynb): BiLSTM and BiLSTM-CRF on NER corpus. 

6 - [Image Classification](https://github.com/syuoni/PyTorch-Tutorials/tree/master/Image-Classification) follows the [PyTorch Image Classification](https://github.com/bentrevett/pytorch-image-classification) repo by [Ben Trevett](https://github.com/bentrevett). 

