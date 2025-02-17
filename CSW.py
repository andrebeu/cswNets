import numpy as np


"""
coffee shop task for neural networks
"""

""" TODO
get_Xfull()
gen_train_data()

write a get_graph which takes in graph_depth and graph_pr 
  and returns the appropriate dict

"""

## data generation parameters

class CSWTask():

  def __init__(self):
    self.end_node = 9
    self.fillerL = [self.end_node+1]

  def get_graph(self,graph_pr):
    """ returns a dict which encodes graph
    {frnode:{tonode:pr,},}
    """
    r = np.round(graph_pr,2)
    s = np.round(1-graph_pr,2)
    graph = {
      0:{1:.5,2:.5},
      1:{3:r,4:s},
      2:{3:s,4:r},
      3:{5:r,6:s},
      4:{5:s,6:r},
      5:{7:r,8:s},
      6:{7:s,8:r},
      7:{9:1.0},
      8:{9:1.0}
    }
    return graph

  def gen_single_path(self,graph):
    """
    begins at 0
    transitions to 1 or 2 w.p. .5
    subsequent transitions controlled by PR
    ends when reach END_NODE
    """
    path = []
    frnode = 0
    while frnode!= self.end_node:
      path.append(frnode)
      distr = graph[frnode]
      tonodes = list(distr.keys())
      pr = list(distr.values())
      frnode = np.random.choice(tonodes,p=pr)
    path.append(self.end_node)
    return np.array(path)

  """ one sentence
  one sentence at a time
  sentence is a filler id and a state id
  """

  def Xfull_onesent(self,fillerL=None,depth=1):
    """
    returns all valid X datapoints 
    """
    X = np.arange(self.end_node)
    F = np.array([np.repeat(filler,len(X)) for filler in fillerL]).reshape(-1)
    X = np.tile(X,len(fillerL))
    X = np.vstack([X,F]).transpose()
    X = slice_and_stride(X,depth)
    return X

  def dataset_onesent(self,path,filler_id,depth=1):
    """ 
    given a path `arr` and filler_id `int`
    returns:
      X = [[[st(t),fi(t)],],]
      Y = [[[st(t+1),f1(t+1)],],]
      shape: (samples,depth,len)
    """
    X = path[1:-2]
    Y = path[2:-1]
    F = np.repeat(filler_id,len(X))
    X = np.vstack([X,F]).transpose()
    Y = np.vstack([Y,F]).transpose()
    X = slice_and_stride(X,depth)
    Y = slice_and_stride(Y,depth)
    return X,Y

  """ 
  full story, no filler marker
  consumes a state at a time, making predictions as it goes
  """

  def Xfull_onestory_det(self):
    """
    only evaluate on high probability paths
    returns X, shape: (samples=2,depth=3,len=1)
    """
    # the first two paths are from 1.0 graph (green context)
    # the second two paths are from 0.0 graph (blue context)
    X = [
        [[1],[3],[5]],
        [[1],[4],[5]],
        [[2],[4],[6]],
        [[2],[3],[6]]
        ]
    return np.array(X)

  def dataset_onestory(self,path,depth=1):
    """ 
    given a path `arr` 
    returns:
      X = [[[st(t)],],]
      Y = [[[st(t+1),],],]
      shape: (samples,depth,len)
    NB each path only generates one sample.
    NB depth == unroll_depth == samples_in_path
    """
    X = path[1:-2]
    Y = path[2:-1]
    X = np.vstack([X]).transpose()
    Y = np.vstack([Y]).transpose()
    X = slice_and_stride(X,depth)
    Y = slice_and_stride(Y,depth)
    return X,Y

  """ full story with graph flag
  as in onestory, but filler_id given as first sample
  the number of input patterns is too large to 
	  make evaluating on all alternatives feasable.
	  instead, I'll change the main loop function
	  to take in a number of paths to evaluate on.
  """

  def dataset_onestory_with_marker(self,path,filler_id,depth=1):
    """ 
    given a path `arr` and filler_id `int`
    returns:
      X = [[[begin,id,st(t),st(t+1)],],]
      Y = [[[id,st(t+1),f1(t+1)],],]
      shape: (samples,depth,len)
    """
    path = np.insert(path,0,filler_id)
    X = path[0:-2]
    Y = path[1:-1]
    X = np.vstack([X]).transpose()
    Y = np.vstack([Y]).transpose()
    X = slice_and_stride(X,depth)
    Y = slice_and_stride(Y,depth)
    return X,Y

  def format_Xeval(self,pathL):
    """
    given a list of paths [[0,10,1,3,5],[0,11,2,4,6]]
    returns an array with format expected by Trainer.predict_step
      (num_samples,depth,in_len)
    """
    Xeval = np.array(pathL)
    Xeval = np.expand_dims(Xeval,2)
    return Xeval



def slice_and_stride(X,depth=1):
  """ 
  useful for including BPTT dim: 
    given (batch,in_len) 
    returns (batch,depth,in_len)
  stride step fixed to = 1
  tf.sliding_window_batch d/n support stride=depth=1
  """
  Xstr = []
  for idx in range(len(X)-depth+1):
    x = X[idx:idx+depth]
    Xstr.append(x)
  return np.array(Xstr)
