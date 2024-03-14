#%%

import numpy as np
from tqdm import tqdm

class LinUCB():
    def __init__(self, n_arms, n_features, alpha = 0.1, random_seed = None):
        np.random.seed(random_seed)
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.eye(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]
        self.theta = np.zeros((self.n_features, 1))
        self.rewards_list = []
        self.rewards_matrix = np.zeros((1, n_arms))
        
    def bandit(self, data, A, t):
        """
        The function bandit() is assumed to take an action and return a corresponding reward.
        """
        rewards_pull = data.copy()[t:t+1][0] 
        R = rewards_pull[A] 
        rewards_pull[:A] = 0
        rewards_pull[A+1:] = 0
        self.rewards_matrix = np.vstack([self.rewards_matrix, rewards_pull.reshape((1,-1))]) 
        self.rewards_list.append(R) 
        
        return R

    def action(self, features):
        p = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta_arm = np.dot(A_inv, self.b[arm])
            p.append( np.dot(theta_arm.T, features) + self.alpha * np.sqrt(np.dot(features.T, np.dot(A_inv, features))) )
        return np.argmax(p)

    def update(self, arm, reward, features):
        self.A[arm] += np.dot(features, features.T)
        self.b[arm] += reward * features.reshape(-1,1)
    
    def train(self, data, features):
        recs = []
        for t in tqdm(range(len(data))):
            
            A = self.action(features[t,:])
            recs.append(A)
            R = self.bandit(data, A, t)
            
            self.update(A, R, features[t,:])
            
        return recs, self.rewards_list, self.rewards_matrix[1:,:]
        
    
            