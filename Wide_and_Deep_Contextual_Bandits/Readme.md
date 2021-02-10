# Contextual Bandit algorithm with Wide and Deep model
A contextual bandit tool compatible with different NN architectures (e.g. wide and deep model with multiple embeddings) and algorithms (LinUCB, Thompson sampling, etc.)

## Wide and Deep model
Wide & Deep learning - jointly trained wide linear models
and deep neural networks - combines the benets of memorization and generalization for recommender systems. -- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) by Cheng et al. (2016)

The Wide_Deep class in this repo gives user a flexible and easy way to define a wide and deep neuron network with multiple embeddings. Please see the instruction [Here](https://github.com/fellowship/space-bandits/blob/dev_tfl/Wide_and_Deep_Contextual_Bandits/Wide_Deep_instruction.ipynb).

## Contextual Bandit
the Contexturalbandit class allows users to plug in their own models and algorithms without changing the code base. Plese see the demo [Here](https://github.com/fellowship/space-bandits/blob/dev_tfl/Wide_and_Deep_Contextual_Bandits/demo_Wide_and_Deep_Contextual_Bandits.ipynb).

## References
[1] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) by Cheng et al. (2016)

[2] [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127) by Riquelme et al. (2018)

[3] [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations](https://arxiv.org/abs/2008.00727) by Guo et al. (2020)
