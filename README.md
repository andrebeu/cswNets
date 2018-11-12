# cswNets: manystories

Beginning Nov 12, 2018

In previous simulations, networks consumed a story at a time. That is, the cell state was flushed to zero after every story.

The current simulations will allow cell states to carry forawrd between different stories. At the beginning of training, cell state will be initialized to zero. I will forward prop and apply learning one story at a time. But I will save the cell state at the end of this forward prop and carry it forward into the next story. 