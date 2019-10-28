Adversarial Anomaly Detection via LSTM
===

Introduction
---

Spatio-temporal event data are becoming increasingly available in a wide variety of applications, such as electronic transaction records, social network data, and crime data. How to efficiently detect anomalies in these dynamic systems using these streaming event data? We propose a novel anomaly detection framework for such event data combining the Long short-term memory (LSTM) and marked spatiotemporal point processes. The detection procedure can be computed in an online and distributed fashion via feeding the streaming data through an LSTM and a neural network-based
discriminator. We study the false-alarm-rate and detection delay using theory and simulation and show that it can achieve weak signal detection by aggregating local statistics over time and networks. Finally, we demonstrate the good performance using real-world datasets.

The major contribution of this work is two-fold: (1) The work has obtained a robust anomaly detector based on a limited amount of training real data. It is proposed to generate ”realistic” fake samples using an adversarial framework to improve the discriminator; (2) The work has proposed modeling the event sequence data by integrating the versatile point process framework with LSTM. This gives the model better interpretability and flexibility in capturing the true underlying pattern.

Our framework is illustrated as follows:
<p align="center"> 
<img src=https://github.com/meowoodie/Adversarial-Anomaly-Detection-via-LSTM/blob/master/imgs/model.pdf width="40%">
</p>
<p align="center"> 
<img src=https://github.com/meowoodie/Adversarial-Anomaly-Detection-via-LSTM/blob/master/imgs/architecture.pdf width="80%">
</p>

Demo
---
`ppgan.py` defines an adversarial model which consists of a discriminator and a generator. Both the discriminator and the generator are constructed by `MSTPP_RNN` object which is a LSTM based sequential data model defined in `pprnn.py`. A training demo is shown below:

```python
from ppgan import PPGAN

with tf.Session() as sess:
    # load data (tensor) which contains multiple sequences
    # each sequence has arbitrary length, the tail of sequence is padded with 1 (for the time) and 0 (for other feature)
    data       = np.load("data/northcal.earthquake.perseason.npy")

    # model configurations
    lstm_hidden_size = 10
    # training configurations
    step_size  = np.shape(data)[1]
    batch_size = 5
    test_ratio = 0.3
    epoches    = 50
    lr         = 1e-2
    n_tgrid    = 50
    n_sgrid    = 50

    # define PPGAN
    ppgan = PPGAN(step_size, lstm_hidden_size, disc_layer_sizes=[20, 10])

    # train via gan
    ppgan.train(sess, batch_size, data, test_ratio, epoches, lr)
```

References
---
- [Shixiang Zhu, Henry Shaowu Yuchi, Yao Xie. "Adversarial Anomaly Detection for Marked Spatio-Temporal Streaming Data."](https://arxiv.org/abs/1910.09161)