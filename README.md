# cswNets: meta learning

Beginning Feb 08, 2019

In previous simulations, networks mostly backproped one story at a time. Eventhough in 'manystories' I carried the cell state forward between different epochs, I continued to only forward and backprop one story at a time. 

Here we setup a metalearning problem, where the network trains in an environment where it receives k stories at a time, and there is a probability that the next story will be from the same versus a different graph. the question we are interested in is whether networks can meta-learn to use prediction error as a signal that the context has changed. some ideas include
- comapre networks that recieve prediction error as input to the cell, against networks that don't: the idea being that networks might be able to compute the prediction error in the cell state once the desired output for the previous timestep is made available as an input as is common in ML. we could even look for whether the network learns to comute such a prediction error.

