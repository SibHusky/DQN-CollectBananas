#Report

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

## Ideas for the future work
- intensive hyperparameter tuning
- a deeper neural network
- other technics like double DQN, dueling DQN, prioritizied experience replay
