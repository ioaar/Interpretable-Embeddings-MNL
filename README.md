# Interpretable-Embeddings-MNL
This repository consists of the source code used in the paper:

Combining Discrete Choice Models and Neural Networks through Embeddings: Formulation, Interpretability and Performance.

## Abstract
This study proposes a novel approach that combines theory and data-driven choice models using Artificial Neural Networks (ANNs). In particular, we use continuous vector representations, called embeddings, for encoding categorical or discrete explanatory variables with a special focus on interpretability and model transparency. Although embedding representations within the logit framework have been conceptualized in [Camara (2019)](https://arxiv.org/abs/1909.00154), their dimensions do not have an absolute definitive meaning, hence offering limited behavioral insights. The novelty of our work lies in enforcing interpretability to the embedding vectors by formally associating each of their dimensions to a choice alternative. Thus, our approach brings benefits much beyond a simple parsimonious representation improvement over dummy encoding, as it provides behaviorally meaningful outputs that can be used in travel demand analysis and policy decisions. Additionally, in contrast to previously suggested ANN-based Discrete Choice Models (DCMs) that either sacrifice interpretability for performance or are only partially interpretable, our models preserve interpretability of the utility coefficients for all the input variables despite being based on ANN principles. The proposed models were tested on two real world datasets and evaluated against benchmark and baseline models that use dummy-encoding. The results of the experiments indicate that our models deliver state-of-the-art predictive performance, outperforming existing ANN-based models while drastically reducing the number of required network parameters.
