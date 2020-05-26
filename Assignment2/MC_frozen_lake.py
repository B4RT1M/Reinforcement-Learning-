# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:13:37 2020

@author: hongh modified by Matthias 
"""

import numpy as np
import gym
import time


def epsilon_greedy(a,env,eps=0.05):
    # Input parma: 'a' : the greedy action for the currently-learned policy  
    # implementation of the epislon-greedy alogrithm.
    # Useful function/variable : np.random.random()/randint() ,  env.nA
    # return the action index for the current state
    
    p = np.random.rand() # First sample a value 'p' uniformly from the interval [0,1), 
    
    # Decide : exploit or explore
    if p < 1-eps:   
        return a 
    else:           
        return np.random.randint(env.nA)

def interact_and_record(env,policy,EPSILON):
    # This function implements the sequential interaction of the agent to environement using decaying epsilon-greedy algorithm for a complete episode
    # It also records the necessary information e.g. state,action, immediate rewards in this episode.
    
    # Initilaize the environment, returning s = S_0
    s = env.reset()     
    state_action_reward = []
    
    # start interaction
    while True:
        a = epsilon_greedy(policy[s],env,eps=EPSILON)
        # Agent interacts with the environment by taking action a in state s,\  env.step()
        # receiving successor state s_, immediate reward r, and a boolean variable 'terminated' telling if the episode terminates.
        # You could print out each variable to check in details.
        s_,r,terminated,_ = env.step(a)
        # store the <s,a,immediate reward> in list for each step
        state_action_reward.append((s,a,r))
        print(r)
        if terminated:            
            break        
        s=s_ 
    
    
    G=0
    state_action_return = []
    state_action_trajectory = []
    
    length = len(state_action_reward)
    # Compute the return G for each state-action pair visited
    # Hint : You need to compute the return in reversed order, first from the last state-action pair, finally to (s_0,a_0)
    
    # gamma = 1 so we let this out 
    
    # it would be easier to calculate the return directly after line 48 but ok.
    #for i in range(length):
    #    G += state_action_reward[length-1-i][2]
        
    # Return : 
    # (1) state_action_return = [(S_(T-1), a_(T-1), G_(T-1)), (S_(T-2), a_(T-2), G_(T-2)) ,... (S_0,a_0.G_0)]
    # 
    # to calculate G backwards we need to substract everytime step reward afterwards
    for j in range(length):
        G += state_action_reward[length-1-j][2]
        state_action_return.append((state_action_reward[length-1-j][0],state_action_reward[length-1-j][1],G)) 
        #G -= state_action_reward[length-1-j][2]
    # (2) state_action_trajectory = [(s_0, a_0), (s_1,a_1), ... (S_(T-1)), a_(T-1))] , note:  the order is different
    # Note: even if (s_n,a_n) is encountered multiple times in an episode, here we still store them in the list, checking if it is the first appearance is done in def monte_carlo()
    for k in range(length): 
        state_action_trajectory.append((state_action_reward[k-1][0],state_action_reward[k-1][1]))
    
    return state_action_return, state_action_trajectory

    
def monte_carlo(env, EPSILON, N_EPISODES):
    # Initialize the random policy , useful function: np.random.choice()  env.nA, env.nS
    policy = np.random.choice(env.nA, env.nS) # Generate a uniform random sample from np.arange(env.nA) of size env.nS (1 x env.nS)
    # Intialize the Q table and number of visit per state-action pair to 0 using np.zeros()
    Q = np.zeros((env.nS,env.nA))
    
    # Initialize a visit per state-action pair to zero 
    visit = np.zeros((env.nS,env.nA))
    
    Returns = np.zeros((env.nS,env.nA))
        
    # MC approaches starts learning
    for i in range(N_EPISODES):
        # Determine the value for epsilon in decaying epsilon-greedy exploration strategy 
        epsilon = 1 - i/N_EPISODES # epsilon = 1 - E_c/E_t 
        # Interact with env and record the necessary info for one episode.
        state_action_return, state_action_trajectory = interact_and_record(env,policy,epsilon)
      
        count_episode_length = 0 # 
        visited = np.zeros((env.nS,env.nA))
        
        for s,a,G in state_action_return:
            count_episode_length += 1
            # Check whether s,a is the first appearnace and perform the update of Q values
            if visited[s,a] == 0:
                Returns[s,a] += state_action_return[count_episode_length-1][2]
                visited[s,a] = 1
                visit[s,a] += 1
                alpha = 1.0 / visit[s,a]
                Q[s,a] = alpha * (Returns[s,a]) 
                # Q[s,a] += alpha * (state_action_return[count_episode_length-1][2] - Q[s,a])
                
                
            # update policy for the current state, np.argmax()
            policy[s] = np.argmax(Q[s,:])# for a in range(env.nA)) # pi(S_t) = argmax_a Q(S_t,a)     
    
    
    
    # Return   the finally learned policy , and the number of visits per state-action pair
    return policy, visit


def run_eps(env, policy, gamma, render):
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


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    
    random_seed = 23333 # Don't change
    N_EPISODES = 150000 # Don't change
    if random_seed:
        env.seed(random_seed)
        np.random.seed(random_seed)
    GAMMA = 1.0
    start = time.time()
    
    policy,visit = monte_carlo(env,EPSILON=1.0,N_EPISODES=N_EPISODES) # I'm not sure about the sense of this EPSILON
    print('TIME TAKEN {} seconds'.format(time.time()-start))
    a2w = {0:'<', 1:'v', 2:'>', 3:'^'}
    # Convert the policy action into arrows
    policy_arrows = np.array([a2w[x] for x in policy])
    # Display the learned policy
    print(np.array(policy_arrows).reshape([-1, 4]))
    
    success = 0 # show how we often our found policy successfully run to the goal
    total_reward, steps = run_eps(env, policy, 1.0, True)
    #for k in range(100):
    #    total_reward, steps = run_eps(env, policy, 1.0, True)
    #    success += total_reward 
        
    print('successful runs out of 100: %d' %success)    
    
    