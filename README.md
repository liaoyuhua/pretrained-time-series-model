# Train a pretrained time series model like BERT

An illustration of how to train a pretrained time series model like BERT.

---
**Principles:**

**Patch**

A single time step contains less information than a word, so we need to segment the time series into patches. This allows us to preserve certain patterns such as fluctuation and local dependency. Although this concept is similar to ViT and MAE, I categorize this project as a sequential modeling one.

**Mask**

Masking is a way to make the model learn the dependency between patches. For example, if we mask the some patches, the model will learn to predict the masked patches based on the unmasked ones. This is similar to the masked language model in BERT. In this way, the model should learn complex representations of the time series in patch level.

**Self-supervised learning**

Generally, this model is trained in a self-supervised way. Although there exists various self-supervised learning methods for time series representation like contrastive learning (TS2Vec and CoST) and transformer-based methods (TST), this project can be regarded as a simple and effective one.

*In a word, you can think of this project as a Patch-BERT for time series.*

---
**Structure:**





---
**References:**

[1] [BERT](https://arxiv.org/abs/1810.04805): One of the most important pretrained language model.

[2] [ViT](https://arxiv.org/abs/2010.11929): Patch-based image model, and introduce the transformer to the computer vision field.

[3] [MAE](https://arxiv.org/abs/2111.06377): Demonstrate the effectiveness of masked version of ViT.

[4] [TS2Vec](https://arxiv.org/abs/2106.10466): Contrastive learning for time series representation.

[5] [CoST](https://arxiv.org/abs/2202.01575): Distangle the time series representation with contrastive learning.

[6] [TST](https://arxiv.org/abs/2010.02803): Classic transformer-based self-supervised learning method for time series, which is highly similar to this project, but it is not powered by patch mask.