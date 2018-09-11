import numpy as np



graph_dict = {
  0:[[1,2],[.5,.5]],
  1:[[3,4],[.8,.2]],
  2:[[3,4],[.2,.8]],
  3:[[5,6],[.8,.2]],
  4:[[5,6],[.2,.8]],
  5:[[7,8],[.8,.2]],
  6:[[7,8],[.2,.8]]
}

def gen_path(graph_dict):
  """ generastes"""
  st = st0 = 0
  path = []
  while st<7:
    node = graph_dict[st]
    next_nodes,pr = node
    next_st = np.random.choice(next_nodes,p=pr)
    st = next_st
    path.append(st)
  return path

def gen_xy(path,filler_id):
  """ given a path through the graph e.g. [A,C] 
  make set of training examples ([A,x],[C,x])
  where A, is state and x is filler_id """
  X = []
  Y = []
  for i in range(len(path)-1):
    x,y = path[i:i+2]  
    X.extend([x,filler_id])
    Y.extend([y,filler_id])
  return np.array(X),np.array(Y)

def gen_m_samples(m,graph_dict,transition_pr,filler_id):
  X = []
  Y = []
  for itr in range(m):
    path = gen_path(graph_dict)
    x,y = gen_xy(path,filler_id)
    X.append(x)
    Y.append(y)
  X = np.array(X)
  Y = np.array(Y)
  return X,Y



def gen_data_dict(story_size=100,RAND=False,COND=False):

  if RAND:
    filler_id = 15
  else:
    filler_id = 9

  if COND:
    X1,Y1 = gen_m_samples(m=int(story_size),graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)
    X2,Y2 = gen_m_samples(m=int(story_size),graph_dict=graph_dict,transition_pr=.2,filler_id=filler_id+1)
    X = np.vstack([X1,X2])
    Y = np.vstack([Y1,Y2])
  else:
    X,Y = gen_m_samples(m=story_size,graph_dict=graph_dict,transition_pr=.8,filler_id=filler_id)

  train_data_dict = {'X_data':X,'Y_data':Y}
  return train_data_dict


