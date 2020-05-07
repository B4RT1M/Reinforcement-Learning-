# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:13:02 2020

@author: hongh
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('FrozenLake-v0')


def value_iter(env, gamma, theta):
    """To Do : Implement Policy Iteration Algorithm
    gamma (float) - discount factor
    theta (float) - termination condition
    env - environment with following required memebers:
        
    Useful variables/functions:
        
            env.nb_states - number of states (AttributeError: 'FrozenLakeEnv' object has no attribute 'nb_states')
            env.nb_action - number of actions
            env.model     - prob-transitions and rewards for all states and actions, you can play around that
            
            The right command are: 
                https://github.com/keerthanpg/FrozenLake-Reinforcement-Learning/blob/master/README.md
            env.nS - number of states
            env.nA - number of actions
            env.P  - transitions, rewards, terminals 
                env.P form: P[s][a] = [(prob, nextstate, reward, is_terminal), ...]
        
        return the value function V and policy pi, 
        pi should be a determinstic policy and an illustration of randomly initialized policy is below
    """
    # Initialize the value function
    V = np.zeros(env.nS)  
    pi = np.zeros(env.nS)
    # to do
    max_it = 100000 # max_iterations 
    for i in range(max_it): 
        prev_V = np.copy(V) # copy of V 
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma * prev_V[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            V[s] = max(q_sa)
            
        if (np.sum(np.fabs(prev_V - V)) <= theta):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break   
        
        pi = compute_pi(env, V, gamma, pi)

#I'm not sure about this 
#            for a in range(env.action_space.n):
#                for next_sr in env.P[s][a]:
#                    # next_sr is a tuple of (probability, next state, reward, done)
#                    p, s_, r, _ = next_sr
#                    q_sa[a] += (p * (r + gamma * V[s_]))
#        pi[s] = np.argmax(q_sa)
    return V, pi

def compute_pi(env, old_V, gamma, pi):
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * old_V[s_]) for p, s_, r, _ in  env.P[s][a]])
        pi[s] = np.argmax(q_sa)
    return pi 

def run_episode(env, policy, gamma, render):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward, step_idx


def discrete_matshow(data, labels_names=[], title=""):
    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin=np.min(data) - .5,
                      vmax=np.max(data) + .5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data), np.max(data) + 1))

    # The names to be printed aside the colorbar
    if labels_names:
        cax.ax.set_yticklabels(labels_names)

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')

if __name__ == '__main__':
    #env.reset()
    #env.render()
    
    gamma = 1.0
    r = 0.
    runs = 100
    
    labels_names=[]
    
    # Check #state, #actions and transition model
    # env.P[state][action]
    print(env.nS, env.nA, env.P[14][2])
        
  
    V, pi = value_iter(env, gamma, theta=1e-4)
    print(V.reshape([4, -1]))
    
    
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([a2w[x] for x in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))
    
    [labels_names.extend([k]) for k in a2w.values()]
    
    discrete_matshow(np.array(pi).reshape([-1, 4]), labels_names, title="policy - value iteration")
    
    
    for k in range(runs):
        total_reward, steps = run_episode(env, pi, gamma, False)
        print('the total reward for this run is %d'%total_reward)
        print('Steps needed to finished the run:  %d.' %steps)
        r += total_reward
        
    print('successful runs out of 100: %d' %r)