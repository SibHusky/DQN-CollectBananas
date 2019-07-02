# Report

## The learning algorithm

The basics of this algorithm is described in http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf and
http://files.davidqiu.com//research/nature14236.pdf

The start values for the hyperparameters were taken from a similar project. 
---
epsilon_start = 1.0  
epsilon_end = 0.01  
epsilon_decay = 0.998 (start value was 0.995, vale was increased for more exploration)  
  
replay buffer size = 10000  
minibatch size = 64  
Gamma = 0.99  
Tau = 0.001  
learning_rate = 0.0005  
Update the network every 4 episodes  

The neural network
---
In this project the state is represented as vector. So a simple 3 layer - DNN is used. (This environment also offers pixel output, then a CNN is needed)
- layer 1: 37  --> 128
- layer 2: 128 --> 128
- layer 3: 128 --> 4

## Result and plots

Episode 100	Average Score: 0.22  
Episode 200	Average Score: 0.48  
Episode 300	Average Score: 1.28  
Episode 400	Average Score: 2.30  
Episode 500	Average Score: 3.09  
Episode 600	Average Score: 4.06  
Episode 700	Average Score: 4.75   
Episode 800	Average Score: 6.27  
Episode 900	Average Score: 6.76  
Episode 1000	Average Score: 7.28  
Episode 1100	Average Score: 8.08  
Episode 1200	Average Score: 8.34  
Episode 1300	Average Score: 9.09  
Episode 1400	Average Score: 9.12  
Episode 1500	Average Score: 9.74  
Episode 1600	Average Score: 10.03  
Episode 1700	Average Score: 11.30  
Episode 1800	Average Score: 11.50  
Episode 1900	Average Score: 11.52  
Episode 2000	Average Score: 10.93  
Episode 2100	Average Score: 11.65    
Episode 2200	Average Score: 12.44  
Episode 2237	Average Score: 13.02  
Environment solved in 2237 episodes.  


## Untrained vs trained agent

| <img src="https://github.com/SibHusky/DQN-CollectBananas/blob/master/gifs/untrained.gif" width="480" height="270" /> | <img src="https://github.com/SibHusky/DQN-CollectBananas/blob/master/gifs/trained_James.gif" width="480" height="270" />  |
|---|---|

## Ideas for the future work
- intensive hyperparameter tuning
- a deeper neural network
- other technics like double DQN, dueling DQN, prioritizied experience replay
