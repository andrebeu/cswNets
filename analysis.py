import numpy as np
from CSW import CSWTask


#
def get_tonodeD(graph):
  """ 
  given a graph 
  return {frnode:[opt,alt]}
    optimal and alternative ids for each frnode
  """
  tonodeD = {0:[1,2]} # by convention
  for frnode,cond_dist in graph.items():
    if frnode == 0: continue 
    arr = np.zeros(2,dtype=np.int)
    for tonode,pr in cond_dist.items():
      if pr>.5: arr[0] = tonode
      elif pr<.5: arr[1] = tonode
    tonodeD[frnode] = arr
  return tonodeD

def avg_over_nodes(yhat_data):
  """ 
  takes pred_data['yhat'] 
    (epochs,path,depth,len,num_classes)
  returns data averaged over frnodes 
    for optimal and alternative node
    (graph_id,epochs,opt_or_alt)
  NB here "optimal node" means 
    node optimal under contextA
  """
  LAST_NODE = 7
  ## nb graph assumption
  task = CSWTask()
  tonodeD = get_tonodeD(task.get_graph(.8))
  # separate data by context
  data = np.stack(
          np.split(
            yhat_data[:,:,:,0,:].squeeze(),
            indices_or_sections=2,
            axis=1))
  # initialize new data array
  new_data = np.empty([*data.shape[:3],3]) # (graphid,epochs,frnode_id,opt_alt_other)
  # only keep predictions to optimal and alternative tonodes
  for frnode_id in range(new_data.shape[2]):
    opt_nodeid,alt_nodeid = tonodeD[frnode_id]
    other_nodeids = np.delete(np.arange(task.end_node+1),[opt_nodeid,alt_nodeid])
    new_data[:,:,frnode_id,0] = data[:,:,frnode_id,opt_nodeid]
    new_data[:,:,frnode_id,1] = data[:,:,frnode_id,alt_nodeid]
    new_data[:,:,frnode_id,2] = np.mean(data[:,:,frnode_id,other_nodeids],axis=2)
    
  # print('sh',new_data.shape)
  # mean over nodes throw out begin and end transitions
  mean_data = np.mean(new_data[:,:,1:LAST_NODE,:],2)
  return mean_data