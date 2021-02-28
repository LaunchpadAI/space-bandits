## Wide and Deep Bandits

A modified version of space-bandits that adds a wide component which takes individual user IDs as inputs. The wide network complements the deep neural network by memorizing the individual user behaviors while the deep network makes generalizations based on the context features. Based on the paper [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations](https://arxiv.org/abs/2008.00727) by Guo et al. (2020).

### Usage

This version retains many of the same features as the original space-bandits. For a demonstration of how the wide and deep model can be used, take a look at the [demo.ipynb](https://github.com/fellowship/space-bandits/blob/dev/wide_deep_bandits/demo.ipynb). 

### Documentation

Please see the [documentation](https://github.com/fellowship/space-bandits/blob/dev/wide_deep_bandits/wide_deep_bandits_documentation.pdf) for more information on the model implementations and parameters. 
