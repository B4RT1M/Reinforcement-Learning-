# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:44:14 2020

@author: barti
"""

import gym
import numpy as np
import json
import statistics
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    with open("SARSA_rewards_q1.json", 'r') as f: 
        SARSA_rewards = json.load(f)

    mean_SARSA = np.zeros((len(SARSA_rewards[0]),1))
    std_SARSA = np.zeros((len(SARSA_rewards[0]),1))
    var_SARSA = np.zeros((len(SARSA_rewards[0]),1))
    
    episodes = np.linspace(0,1999,num=2000)#np.arange(0,2000, 1)
    temp_S = np.zeros((len(SARSA_rewards),1))
    
    for i in range(len(SARSA_rewards[0])):
        for j in range(len(SARSA_rewards)):
            temp_S[j,0] = SARSA_rewards[j][i]
        
        mean_SARSA[i,0] = np.mean(temp_S, axis = 0)
        std_SARSA[i,0] = np.std(temp_S, axis = 0)
        var_SARSA[i,0] = np.var(temp_S, axis = 0)
        
    
    y1_SARSA = mean_SARSA - std_SARSA
    #y1_SARSA = np.array(y1_SARSA,dtype = np.float64)
    y1_SARSA = np.array(y1_SARSA).reshape([2000])
    y2_SARSA = mean_SARSA + std_SARSA
    y2_SARSA = np.array(y2_SARSA).reshape([2000])
    #y2_SARSA = np.array(y2_SARSA,dtype = np.float64)
    
    
    
    #plt.ylim(-1000,0)
    plt.plot(episodes, mean_SARSA, color='b', alpha=1, label='pop = 50')
    plt.fill_between(episodes, y1_SARSA, y2_SARSA, color='b', alpha=0.2, label ='95% CI')   
    plt.title(r'MountainCar-v0 - SARSA($\lambda$)')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend(('SARSA (mean)', 'SARSA (std)'))
    #plt.legend(('SARSA (mean)', 'Q-Learning (mean)'))
    plt.savefig('graphic_task2_q1.svg')
    plt.show()    
        
        
        