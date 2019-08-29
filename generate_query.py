import numpy as np
import pandas as pd
import GPyOpt
import GPy 
import pickle

import util
from util import X_names, X_bounds, get_domain
from util import rescale

def initial_design(bounds, batchsize, method="latin"):
    dim = len(bounds)
    x = np.random.uniform(size=[batchsize,dim])
    if method=="latin":
        for i in range(0,dim):
            x[:,i] = ((np.argsort(x[:,i])+np.random.uniform(size=[batchsize]))/batchsize)*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
    elif method=="uniform":
        for i in range(0,dim):
            x[:,i] = x[:,i]*(bounds[i][1]-bounds[i][0]) + bounds[i][0]
    return np.round(x, -1)

def calc_EIstep(X_init, Y_init, batchsize, normalize, savepath, kernel):
    space = GPyOpt.core.task.space.Design_space(get_domain(normalize=normalize), None)
    if kernel == "RBF":
        model_gp = GPyOpt.models.GPModel(kernel=GPy.kern.RBF(input_dim=X_init.shape[1], ARD=True),ARD=True,verbose=False)
    elif kernel == "matern52":
        model_gp = GPyOpt.models.GPModel(ARD=True, verbose=False)
    objective = GPyOpt.core.task.SingleObjective(None)
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    acquisition_EI = GPyOpt.acquisitions.AcquisitionEI(model_gp, space, acquisition_optimizer, jitter=0)
    acquisition = GPyOpt.acquisitions.LP.AcquisitionLP(model_gp, space, acquisition_optimizer,acquisition_EI)
    evaluator = GPyOpt.core.evaluators.LocalPenalization(acquisition, batch_size=batchsize)
    
    bo_EI = GPyOpt.methods.ModularBayesianOptimization(
    model=model_gp,
    space=space,
    objective=objective,
    acquisition=acquisition,
    evaluator=evaluator,
    X_init=X_init,
    Y_init=Y_init,   
    normalize_Y=True
    )
    
    nextX = bo_EI.suggest_next_locations()
        
    if normalize:
        nextX = rescale(nextX)
    
    with open( savepath+"/model/EI_j"+".pkl", "wb") as f:
        pickle.dump(bo_EI, f, protocol=2)
    
    return nextX