import numpy as np


"""
coffee shop task for neural networks
"""

"""
todo make into class implementation
"""

## data generation parameters
END_NODE = 9
GRAPH_DEPTH = 4
GRAPH_PR = 0.8

class CSW():
  def __init__(self):
    self.end_node = 9
    self.graph_depth = 4
    self.graph_pr = 0.8

  def get_graph(self,graph_pr):
    """ returns a dict which encodes graph
    {frnode:{tonode:pr,},}
    """
    graph = {
      0:{1:.5,2:.5},
      1:{3:graph_pr,4:(1-graph_pr)},
      2:{3:(1-graph_pr),4:graph_pr},
      3:{5:graph_pr,6:(1-graph_pr)},
      4:{5:(1-graph_pr),6:graph_pr},
      5:{7:graph_pr,8:(1-graph_pr)},
      6:{7:(1-graph_pr),8:graph_pr},
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
    return path

  def gen_path_from_len(self,path_len,graph):
    """ 
    generates a path of path_len by combining multiple single_paths
    """
    paths = []
    num_paths = int(path_len/self.graph_depth)+2
    for _ in range(num_paths):
      path = gen_single_path(graph)
      paths.extend(path[1:-1])
    return paths[:path_len]

  def gen_path_from_numstories(self,num_stories,graph):
    """ 
    generastes path containing num_stories passess through graph
    """
    return None

  def gen_dataset(self,graph_pr,path_len):
    """ 
    returns data to be fed to neural network
    """
    graph = self.get_graph(graph_pr)
    path = self.gen_path_from_numstories(path_len,graph)
    X = np.array([path[:-1]]).transpose()
    Y = np.array([path[1:]]).transpose()
    return X,Y



