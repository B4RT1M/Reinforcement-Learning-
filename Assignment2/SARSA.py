#!/usr/bin/env/ python
"""
@author hongh modified by Matthias
"""
import gym
import numpy as np
import json

# https://www.geeksforgeeks.org/sarsa-reinforcement-learning/

class SARSA_Learner(object):
    def __init__(self, env):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins_d1 = 20  # Number of bins to Discretize in dim 1
        self.obs_bins_d2 = 15    # Number of bins to Discretize in dim 2 -velocity, should be odd number!
        self.obs_bins = np.array([self.obs_bins_d1,self.obs_bins_d2])
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n

        # Q-values, Initialize the Q table with -1e-7 , in the last task, you can initialize it with 0 and compare the results, for task III and question III with alpha = 1/#visit
        self.Q = np.ones((self.obs_bins[0] + 1, self.obs_bins[1] + 1, self.action_shape)) * (-1e-7)#0 #(21 x 16 x 3)
        # Initialize the visit_counts
        self.visit_counts = np.zeros((self.obs_bins[0] + 1, self.obs_bins[1] + 1, self.action_shape))
        self.alpha = 0.05  # Learning rate
        self.gamma = 1.0  # Discount factor
        self.epsilon = 1.0 # Initialzation of epsilon value in epsilon-greedy

    def discretize(self, obs):
        '''A function maps the continuous state to discrete bins
        '''
        return tuple(((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, state):
        # dicreteize the observation first
        discretized_state = self.discretize(state)
        ''' 
            Implement the behavior policy (episilon greedy policy)
            return the discrete action index
        '''
        action = 0
        p = np.random.uniform(0, 1) # First sample a value 'p' uniformly from the interval [0,1),
        
        # Decide : exploit or explore 
        if p < 1 - self.epsilon:
            action =  np.argmax(self.Q[discretized_state[0],discretized_state[1],:])
        else:
            action =  env.action_space.sample()
        
        return action


    def update_Q_table(self, obs, action, reward, done, next_obs, action2):
        '''
           update the Q table self.Q given each state,action ,reward... 
           No parameters for return
           Directly update the self.Q here and other necessary variables here.
        '''
        discretized_state1 = self.discretize(obs)
        discretized_state2 = self.discretize(next_obs)
        
        predict = self.Q[discretized_state1[0], discretized_state1[1], action]
        target = reward + self.gamma * self.Q[discretized_state2[0], discretized_state2[1], action2]
        self.Q[discretized_state1[0],discretized_state1[1], action] = self.Q[discretized_state1[0],discretized_state1[1], action] + self.alpha * (target - predict)

        
def train(agent, env, MAX_NUM_EPISODES,max_eps_length):
    ''' 
        Implement one step Q-learning algorithm with decaying epsilon-greedy explroation and plot the episodic reward w.r.t. each training episode
        
        return: (1) policy, a 2-dimensional array, it is 2 dimensional since the state is 2D. Each entry stores the learned policy(action).
                (2) Q table, a 3-D array
                (3) Number of visits per state-action pair, 3D array
        Useful functions: env.step() , env.reset(), 
        Recommended : print the episodic reward per episode to check you are writing the program correctly. You can also track the best episodic reward until so far
    '''
    best_reward = -float('inf')
    policy = np.zeros((agent.Q.shape[0],agent.Q.shape[1]))
    rewards = []
    
    for episode in range(MAX_NUM_EPISODES):
        # update the epsilon for decaying epsilon-greedy exploration
        agent.epsilon = 1 - episode/MAX_NUM_EPISODES
        # initialize the state
        obs = env.reset()
        # (1) Select an action for the current state
        action1 = agent.get_action(obs)
        # initialization of the following variables
        done = False
        k = 0
        
        total_reward = 0.0
        # To complete: one complete episode loop here.
        # (1) Select an action for the current state, using  agent.get_action(obs)
        # (2) Interact with the environment, get the necessary info
        # (3) Update the Q tables using agent.update_Q_table()
        # (4) also record the episodic cumulative reward 'total_reward'
        # (5) Update the visit_counts per state-action pair
        while k < max_eps_length:    
            
            # (2) Interact with the environment 
            obs2, reward, done, info = env.step(action1)
            
            # choose the next action for obs2
            action2 = agent.get_action(obs2)
            
            # (3) Update the Q tables 
            agent.update_Q_table(obs, action1, reward, done, obs2, action2)
            
            # (4) record the episodic cumulative reward 
            total_reward += reward
            
            # (5) Update the visit_counts per state action pair
            d_state = agent.discretize(obs)
            
            agent.visit_counts[d_state[0],d_state[1], action1] = agent.visit_counts[d_state[0],d_state[1], action1] + 1
            
            obs = obs2
            action1 = action2
            
            k += 1
            
            # if we reached the end of the learning process: 
            if done: 
                print('The final reward are: %d' %total_reward)
                break 
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode, 
                                     total_reward, best_reward, agent.epsilon))
        # Return the trained policy
        rewards.append(total_reward)
        for s in range(agent.obs_bins[0]):
            for i in range(agent.obs_bins[1]):
                policy[s,i] = np.argmax(agent.Q[s,i,:])
                
    return policy, agent.Q.copy(), agent.visit_counts.copy(), rewards


def test(agent, env, policy, max_eps_length, render):
    ''' 
        test the agent with the learned policy, the structure is very similar to train() function.
        In the test phase, we choose greedy actions, we don;t update the Q-table anymore.
        Return : episodic reward (cumulative reward in an episode)
        Constrain the maximal episodic length to be 1000.
        for local users : you can add additional env.render() after env.step(a) to see the trained result.
    '''
    episodic_reward = 0
    agent.epsilon = 0 
    k = 0
    
    obs = env.reset()
    while k < max_eps_length :
        if render: 
            env.render()
        # (1) Select an action for the current state
        action = agent.get_action(obs)
            
        # (2) Interact with the environment 
        obs2, reward, done, info = env.step(action)
        
        # (4) record the episodic cumulative reward 
        episodic_reward += reward
            
        
        obs = obs2
        k += 1
        if done :
            break
    return episodic_reward

if __name__ == "__main__":
    ''' 
    TODO : You need to add code for plotting the result.
    '''
    MAX_NUM_EPISODES = 2000 
    env = gym.make('MountainCar-v0').env     # Note: the episode only terminates when cars reaches the target, the max episode length is not clipped to 200 steps.
    agent = SARSA_Learner(env)
    max_eps_length = 1000
    rewards_4_plot = []
    
    for _ in range(5):
            
        learned_policy, Q, visit_counts, rewards = train(agent, env, MAX_NUM_EPISODES,max_eps_length)
        rewards_4_plot.append(rewards)
        
    # Display the learned policy
    #print(learned_policy)
    
    # after training, test the policy 10 times.
    #for _ in range(10):
    #    reward = test(agent, env, learned_policy, max_eps_length, True)
    #    print("Test reward: {}".format(reward))
    env.close()
    
    with open("SARSA_rewards.json", "w") as f:
        json.dump(rewards_4_plot, f, indent = 2)