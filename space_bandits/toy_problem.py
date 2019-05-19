import numpy as np
import pandas as pd
from random import random, randint

def get_customer(ctype=None):
    """Customers come from two feature distributions.
    Class 1: mean age 25, var 5 years, min age 18
             mean ARPU 100, var 15
    Class 2: mean age 45, var 6 years
             mean ARPU 50, var 25
    """
    if ctype is None:
        if random() > .5: #coin toss
            ctype = 1
        else:
            ctype = 2
    age = 0
    ft = -1
    if ctype == 1:
        while age < 18:
            age = np.random.normal(25, 5)
        while ft < 0:
            ft = np.random.normal(100, 15)
    if ctype == 2:
        while age < 18:
            age = np.random.normal(45, 6)
        while ft < 0:
            ft = np.random.normal(50, 25)
    age = round(age)
    return ctype, (age, ft)

def get_rewards(customer):
    """
    There are three actions:
    promo 1: low value. 10 dollar if accept
    promo 2: mid value. 25 dollar if accept
    promo 3: high value. 100 dollar if accept

    Both groups are unlikely to accept promo 2.
    Group 1 is more likely to accept promo 1.
    Group 2 is slightly more likely to accept promo 3.

    The optimal choice for group 1 is promo 1; 90% acceptance for
    an expected reward of 9 dollars each.
    Group 2 accepts with 25% rate for expected 2.5 dollar reward

    The optimal choice for group 2 is promo 3; 20% acceptance for an expected
    reward of 20 dollars each.
    Group 1 accepts with 2% for expected reward of 2 dollars.

    The least optimal choice in all cases is promo 2; 10% acceptance rate for both groups
    for an expected reward of 2.5 dollars.
    """
    if customer[0] == 1: #group 1 customer
        if random() > .1:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .98:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 2: #group 2 customer
        if random() > .75:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .80:
            reward3 = 100
        else:
            reward3 = 0
    return np.array([reward1, reward2, reward3])

def get_cust_reward():
    """returns a customer and reward vector"""
    cust = get_customer()
    reward = get_rewards(cust)
    fts = cust[1]
    return np.array([fts])/100, reward

def generate_dataframe(n_rows):
    df = pd.DataFrame()
    ages = []
    ARPUs = []
    actions = []
    rewards = []
    for i in range(n_rows):
        cust = get_customer()
        reward_vec = get_rewards(cust)
        context = np.array([cust[1]])
        ages.append(context[0, 0])
        ARPUs.append(context[0, 1])
        action = np.random.randint(0,3)
        actions.append(action)
        reward = reward_vec[action]
        rewards.append(reward)

    df['age'] = ages
    df['ARPU'] = ARPUs
    df['action'] = actions
    df['reward'] = rewards

    return df

def get_customer_nl(ctype=None):
    """Customers come from two feature distributions.
    Class 1: mean age 25, var 5 years, min age 18
             mean ARPU 100, var 15
    Class 2: mean age 45, var 6 years
             mean ARPU 50, var 25
    """
    if ctype is None:
        ctype = randint(0,2)
    age = 0
    ft = -1
    if ctype == 0:
        while age < 18:
            age = np.random.normal(25, 5)
            ft = 125 - .1*(age-25)*(age-25) + np.random.normal(0, 4)
    if ctype == 1:
        while age < 18:
            age = np.random.normal(35, 2)
        while ft < 0:
            ft = np.random.normal(75, 3)
    if ctype == 2:
        while age < 18:
            age = np.random.normal(45, 6)
            ft = 25 + .25*(age-45)*(age-45) + np.random.normal(0, 4)
    age = round(age)
    return ctype, (age, ft)

def get_rewards_nl(customer):
    """
    There are three actions:
    promo 1: low value. 10 dollar if accept
    promo 2: mid value. 25 dollar if accept
    promo 3: high value. 100 dollar if accept

    Expected Value Matrix:

           group1|group2|group3
    ----------------------------
    promo1|  $9  |  $1  |  $1
    ----------------------------
    promo2| $2.5 | $12.5| $1.25
    ----------------------------
    promo3| $1   |  $5  | $25

    We can see each group has an optimal choice.
    """
    if customer[0] == 0: #group 1 customer
        if random() > .1:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .90:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .99:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 1: #group 2 customer
        if random() > .9:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .50:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .95:
            reward3 = 100
        else:
            reward3 = 0
    if customer[0] == 2: #group 3 customer
        if random() > .9:
            reward1 = 10
        else:
            reward1 = 0
        if random() > .95:
            reward2 = 25
        else:
            reward2 = 0
        if random() > .75:
            reward3 = 100
        else:
            reward3 = 0
    return np.array([reward1, reward2, reward3])

def get_cust_reward_nl():
    """
    returns a customer and reward vector
    for nonlinear problem
    """
    cust = get_customer()
    reward = get_rewards(cust)
    age = cust[1]
    return np.array([age]), reward

def generate_biased_dataframe(n_rows):
    df = pd.DataFrame()
    ages = []
    ARPUs = []
    actions = []
    rewards = []
    for i in range(n_rows):
        cust = get_customer()
        reward_vec = get_rewards(cust)
        context = np.array([cust[1]])
        age = context[0, 0]
        ARPU = context[0, 1]
        ages.append(age)
        ARPUs.append(ARPU)
        if ARPU <= 50:
            action = 0
        elif ARPU <= 100:
            action = 1
        else:
            action = 2
        actions.append(action)
        reward = reward_vec[action]
        rewards.append(reward)

    df['age'] = ages
    df['ARPU'] = ARPUs
    df['action'] = actions
    df['reward'] = rewards

    return df
