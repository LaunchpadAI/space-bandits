## Wide and Deep Bandits

A modified version of space-bandits that adds a wide component that can take individual user IDs as inputs. Based on the paper [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations](https://arxiv.org/abs/2008.00727) by Guo et al. (2020).The wide network complements the deep neural network by memorizing the individual user behaviors while the deep network makes generalizations based on the context features. 

### Usage

This version retains many of the same functions as space-bandits. For a demonstration of how the wide and deep model can be used, talk a look at the [demo.ipynb](https://github.com/fellowship/space-bandits/blob/dev/wide_deep_bandits/demo.ipynb)
