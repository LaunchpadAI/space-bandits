## Wide and Deep Bandits

A modified version of space-bandits that adds a wide component that can take individual user IDs as inputs. Based on the paper [Deep Bayesian Bandits: Exploring in Online Personalized Recommendations](https://arxiv.org/abs/2008.00727) by Guo et al. (2020).The wide network complements the deep neural network by memorizing the individual user behaviors while the deep network makes generalizations based on the context features. 

### Usage

This version retains many of the same features as the original space-bandits. For a demonstration of how the wide and deep model can be used, talk a look at the [demo.ipynb](https://github.com/fellowship/space-bandits/blob/dev/wide_deep_bandits/demo.ipynb)

### Action Selection

This version uses Bayesian Linear Regression and Thompson Sampling similar to the original space-bandits. Action can be selected using one of the following methods.

- **BLR**: Use the expected rewards from the Bayesian linear regression to predict best action. 

- **BLR+TS**: Rewards are sampled from the posterior distribution using Thompson Sampling.  

- **Forward**: Use the predicted rewards from the neural networks to select the actions directly. 

### Wide and Deep Model Combination Method

The wide network trains an embedding using the user IDs, and the deep network consist of multiple linear layers. The wide and deep components can be combined using several different methods. 

- **Add Rewards**: The predicted rewards from the wide and deep network are added in the last layer and loss is computed on the sum. 

- **Concatenate Rewards**: The predicted rewards from the wide and deep network are concatenated and loss is computed on the concatenation. 

- **Linear Combination of Rewards**: The predicted rewards from the wide and deep network are combined using a final linear layer. 

- **Last Layer Representation**: The embedding from the wide model is concatenated with the last layer representation of the deep neural network. A final linear layer then uses the concatenated representations to predict the reward. 

