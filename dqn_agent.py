import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import namedtuple, deque

import random

#cpu or gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class QNetwork(nn.Module):
    """
    The DNN for predict an action
    """
    def __init__(self,state_size, action_size, list_number_units_hidden,seed):
        """
        Parameters
        ==========
            state_size (int): Dimension of each state (input features)
            list_number_units_hidden list[int]: Every entry is the number of nodes of a hidden layer
                example:    list_number_units_hidden = [4,3]
                            means the nn has 2 hidden layers. The first with 4 and the second with 3 nodes
            action_size (int): Number of actions (output)
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        """
        Initialize model
        """
        self.seed = torch.manual_seed(seed)
        l = []
        l.append(state_size)
        for hidden in list_number_units_hidden:
            l.append(hidden)
        l.append(action_size)
        self.modelList = nn.ModuleList()
        for layer in range(len(l)-1):
            self.modelList.append(nn.Linear(l[layer],l[layer+1]))

    def forward(self, state):
        """
        Build the feed forward network
        Parameters
        ==========
            state (vector): state_size x batch_size

        Returns:
            vector (action_size x batch_size)
        """
        for lin in self.modelList:
            if lin == self.modelList[-1]:
                state = lin(state)
            else:
                state = F.relu(lin(state))

        return state



class ReplayBuffer:
    """
    Fixed-size Buffer for the experience tuples
    """
    def __init__(self,maxlen,batch_size):
        """
        Parameters
        ==========
        maxlen(int): The max number of entrys in the ring buffer
        batch_size(int): the size of the batch that is given back by the getbatch function 
        """
        self.rb = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.expi = namedtuple("Expi", ["state","action","reward","next_state","done"])

    def add(self,state,action,reward,next_state,done):
        """Add a new experience tuple to the ring buffer"""
        single_expi = self.expi(state,action,reward,next_state,done)
        self.rb.append(single_expi)

    def getbatch(self):
        """
        returns a randomly sample batch with the size of batch_size for feeding the DNN
        includes convertation to torch tensor

        """
        experience = random.sample(self.rb, k=self.batch_size)
        torch_states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(device)
        torch_actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(device)
        torch_rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(device)
        torch_next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(device)
        torch_done = torch.from_numpy(np.vstack([1 if e.done == True else 0 for e in experience if e is not None])).float().to(device)
    
        return (torch_states, torch_actions, torch_rewards, torch_next_states,torch_done)


    def __len__(self):
        return len(self.rb)


class Agent():
    """
    """
    def __init__(self,state_size,action_size,list_number_units_hidden,seed,learn_rate,LearningUpdateEvery,maxlen,tau,filename=None):
        """
        Interaction with the environment 
        Parameters
        ==========
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            list_number_units_hidden list[int]: Every entry is the number of nodes of a hidden layer
                example:    list_number_units_hidden = [4,3]
                            means the nn has 2 hidden layers. The first with 4 and the second with 3 nodes
            seed (int): random seed
            learn_rate (int): learning rate for the Adam optimizer
            LearningUpdateEvery (int): Update rate for the network
            maxlen (int): The max number of entrys in the ring buffer
            tau (int): for the softupdate
            filename (string): filename of the trained weights

        """
        self.state_size = state_size
        self.action_size = action_size
        self.list_number_units_hidden = list_number_units_hidden

        self.tau = tau
        self.UpdateCounter = 0

        # DQN
        self.Q_target = QNetwork(self.state_size,self.action_size,self.list_number_units_hidden,seed).to(device)
        self.Q_local = QNetwork(self.state_size,self.action_size,self.list_number_units_hidden,seed).to(device)
        #the two DNN must have the same weights
        #self.Q_target.load_state_dict(self.Q_local.state_dict())
        self.optimizer = optim.Adam(self.Q_local.parameters(),lr=learn_rate)

        if filename:
            trained_weights = torch.load(filename)
            self.Q_local.load_state_dict(trained_weights)
            self.Q_target.load_state_dict(trained_weights)

        # Replay Memory
        self.batch_size = 64
        self.RepMem = ReplayBuffer(maxlen,self.batch_size)
        self.LearnStepCounter = 0
        self.LearningUpdateEvery = LearningUpdateEvery

    def step(self,state,action,reward,next_state,done,gamma):
        """
        add an entry in replay memory
        Learn every x*step if enough samples are available
        """
        self.RepMem.add(state,action,reward,next_state,done)
        self.LearnStepCounter = (self.LearnStepCounter + 1) % self.LearningUpdateEvery
        if self.LearnStepCounter  == 0:
            if len(self.RepMem) >= self.batch_size:
                #learn
                batch = self.RepMem.getbatch()
                self.learn(batch,gamma,self.tau)


    def act(self,state,eps):
        """
        Returns the action for the current state
        """
        ##calculate current eps
        #self.eps_current = max(self.eps_current*self.eps_decay,self.eps_min)
        
        #greedy or random action
        if eps < random.random():
            #use greedy action, the action predicted from the DQN
            #feed forward, set eval mode
            self.Q_local.eval()
            #convert state into torch tensor and send to device(cuda)
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # speed things up: no grad
            with torch.no_grad():
                action = self.Q_local(state)
            self.Q_local.train()
            return np.argmax(action.cpu().numpy()).astype(int)
            #return np.argmax(action.cpu().data.numpy())

        else:
            #use random action
            return random.choice(np.arange(self.action_size))




    def learn(self,batch,gamma,tau):
        """
        Do the Backprob
        """
        torch_states, torch_actions, torch_rewards, torch_next_states, torch_dones =  batch
            
        
        #"the ground-truth", the targets weights musst be treat as constants, no backprob. detach is used. 
        Q_target_next = self.Q_target(torch_next_states).detach().max(1)[0].unsqueeze(1)
        #choose the max values with max(1), use [0] to get only the values(without the indices)
        target = torch_rewards + (gamma * Q_target_next * (1-torch_dones))
        #the prediction. choose the value for torch_action. use gather function
        prediction = self.Q_local(torch_states).gather(1,torch_actions.long())
        #error function
        
        loss = F.mse_loss(prediction,target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.Q_local,self.Q_target,tau)

        #Update weights of Q_target
        # self.UpdateCounter += 1
        # if self.UpdateCounter < 0:
        #     for local_weights, target_weights in zip(self.Q_local.parameters(),self.Q_target.parameters()):
        #         target_weights.data.copy_(local_weights.data)
        # else:
        # #Update weights of Q_target soft
        #     for local_weights, target_weights in zip(self.Q_local.parameters(),self.Q_target.parameters()):
        #         target_weights.data.copy_(tau*local_weights.data + (1.0-tau)*target_weights.data)

    def soft_update(self, local_model, target_model, tau):
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

if __name__ == "__main__":
    print ("main loop")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print (device)

