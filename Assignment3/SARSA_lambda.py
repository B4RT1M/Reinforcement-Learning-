"""
    @author hongh modified by Matthias
"""

#!/usr/bin/env/ python
import gym
import numpy as np
import json
import time

def accumulating_trace(trace, activeTiles, lam, gamma):
    '''
        accumulating trace
            # trace: old trace
            # activeTiles: current active tile indices
            # lam: lambda
            # return: updated trace
    '''
    trace *= gamma * lam
    trace[activeTiles] += 1
    
    
def replacing_trace(trace, activeTiles, lam, gamma):
    '''
        replacing trace update rule
            # trace: old trace 
            # activeTiles: current active tile indices
            # lam: lambda
            # return: updated trace
    '''
    # update the trace here.
    trace *= gamma * lam
    trace[activeTiles] = 1
    return trace



class SARSA_lambda_Learner(object):
    def __init__(self, env, traceUpdate):
        self.obs_shape = env.observation_space.shape
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.obs_bins_d1 = 20 # Number of bins to Discretize in dim 1
        self.obs_bins_d2 = 15    # Number of bins to Discretize in dim 2
        self.obs_bins = np.array([self.obs_bins_d1,self.obs_bins_d2])
        self.bin_width = (self.obs_high - self.obs_low) / self.obs_bins
        self.action_shape = env.action_space.n
       
        # TODO : modify the inital Q-values
        self.Q = np.ones((self.obs_bins[0]+1, self.obs_bins[1]+1, self.action_shape)) * 0
        self.visit_counts = np.zeros((self.obs_bins[0]+1, self.obs_bins[1]+1, self.action_shape))
        self.alpha = 0.05  # Learning rate
        self.gamma = 1  # Discount factor
        self.epsilon = 1
        # Initialize the trace for each discretized state-action pair
        # with d = 21x16x3 = 1008
        self.trace = np.zeros((self.obs_bins[0]+1,self.obs_bins[1]+1,self.action_shape))
        self.traceUpdate = traceUpdate 
        self.lam = 0.95 # lambda, to be modified in other tasks
        
    def discretize(self, obs):
        return tuple(np.round((obs - self.obs_low) / self.bin_width).astype(int))

    def get_action(self, obs):
        discretized_obs = self.discretize(obs)
        # Epsilon-Greedy action selection
        if np.random.random() > self.epsilon: # Exploit
            return np.argmax(self.Q[discretized_obs])
        else:  # Explore
            return np.random.choice([a for a in range(self.action_shape)])

    def learn(self, obs, action, reward, done, next_obs, next_action):
        discretized_obs = self.discretize(obs)
        discretized_next_obs = self.discretize(next_obs)
        discretized_sa = discretized_obs + (action,)
        discretized_next_sa = discretized_next_obs + (next_action,)
        
        # Compute td_delta
        if done == True:
            td_delta = reward - self.Q[discretized_sa]
        else:    
            td_delta = reward + self.gamma * self.Q[discretized_next_sa] - self.Q[discretized_sa]
        
        # Update the visit counts as statistics for later analysis.
        self.visit_counts[discretized_obs][action] += 1
        # Update the trace
        if self.traceUpdate == replacing_trace: # self.traceUpdate == accumulating_trace
            self.traceUpdate(self.trace, discretized_sa, self.lam, self.gamma)
        else:
            raise Exception('Undefined Trace')
        
        # update the q table
        self.Q[discretized_sa] = self.Q[discretized_sa] + self.alpha*td_delta*self.trace[discretized_sa]
        
        # post-process the trace after the episode ends
        # unsure : we need to set the trace to zero again
        if done == True:            
            self.trace = np.zeros((self.obs_bins[0]+1,self.obs_bins[1]+1,self.action_shape)) 


def train(agent, env, MAX_NUM_EPISODES, t2_q2):
    best_reward = -float('inf')
    # Record the total number of interactions
    total_interaction_count = 0
    policy = np.zeros((agent.Q.shape))
    rewards = []
    for episode in range(MAX_NUM_EPISODES):
        
        step = 0
        # Decaying epsilon-greedy
        agent.epsilon = 1 - episode/MAX_NUM_EPISODES
        
        done = False
        # Fetch the initial state
        obs = env.reset()
        episodic_reward = 0.0 # episodic reward
        action = agent.get_action(obs)
        while not done:            
            next_obs, reward, done, info = env.step(action)
            total_interaction_count+= 1
            step += 1
            next_action = agent.get_action(next_obs)
            agent.learn(obs, action, reward, done, next_obs, next_action)     
            episodic_reward += reward
            action = next_action
            obs = next_obs
            
            if t2_q2 == True and done: #step >= 5000 and 
                agent.trace = np.zeros((agent.obs_bins[0]+1,agent.obs_bins[1]+1,agent.action_shape))
                break
        rewards.append(episodic_reward)    
        if episodic_reward > best_reward:
            best_reward = episodic_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
                                     episodic_reward, best_reward, agent.epsilon))
    # Display the total number of interactions (as one measure of convergence speed)
    print('Total number of interactions: {}'.format(total_interaction_count))
    
    
    # Return the trained policy as an 2D-array
    for s in range(agent.obs_bins[0]+1):
        for i in range(agent.obs_bins[1]+1):
            policy[s,i] = np.argmax(agent.Q[s,i,:])
    return policy, agent.Q.copy(), agent.visit_counts.copy(), rewards


def test(agent, env, policy):
    done = False
    obs = env.reset()
    episodic_reward = 0
    step_count = 0
    while not done:
        env.render()
        action = policy[agent.discretize(obs)]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        episodic_reward += reward
        step_count += 1
        if step_count >= 1000: # Force stopping after 1000 steps to avoid infinite loops
            break
    return episodic_reward


    

if __name__ == "__main__":
    ''' 
        TODO : You need to add code for plotting the result and saving the statistics as in the last assignment.
    '''
    # TODO: later modify the MAX_NUM_EPISODES
    MAX_NUM_EPISODES = 5000 
    env = gym.make('MountainCar-v0').env     # Note: the episode only terminates when cars reaches the target, the max episode length is not clipped to 200 steps.
    agent = SARSA_lambda_Learner(env, replacing_trace)
    Q_runs = []
    times = []
    rewards_4_plot = []
    t2_q2 = True
    
    for i in range(5):
        print(i)
        if t2_q2 == True: 
            agent.Q = np.ones((agent.Q.shape)) * 0
            if i == 0:
                agent.lam = 0.99
            elif i == 1: 
                agent.lam = 0.95
            elif i == 2: 
                agent.lam = 0.75
            elif i == 3: 
                agent.lam = 0.5
            elif i == 4: 
                agent.lam = 0   
                
        start = time.time()    
        learned_policy, Q, visit_counts, reward = train(agent, env, MAX_NUM_EPISODES, t2_q2)
        end = time.time()-start
        #np.save('SARSA_' + str(agent.lam) + '_10000' + '.npy',Q)
        times.append(end)
        Q_runs.append(Q)
        rewards_4_plot.append(reward)
    # save the Q-table here
    if t2_q2 == True:
        np.save('SARSA_q2_Q0_maxInf_eps5000.npy',Q_runs)
        np.save('SARSA_times_q2_Q0_maxInf_eps5000.npy', times)
    else: 
        np.save('SARSA_' + str(agent.lam) + '.npy',Q_runs)
        np.save('SARSA_times.npy', times)
        with open("SARSA_rewards_q1.json", "w") as f:
            json.dump(rewards_4_plot, f, indent = 2)
     
    # after training, test the policy 10 times.
    # test doesn't work (CHECK) 
    # action = policy[agent.discretize(obs)]
    #for _ in range(10):
    #    reward = test(agent, env, learned_policy)
    #    print("Test reward: {}".format(reward))
    env.close()
    #with open("SARSA_rewards_q2.json", "w") as f:
    #    json.dump(Q_runs, f, indent = 2)