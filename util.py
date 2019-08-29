import numpy as np
import pandas as pd

upper = 500.

X_names = [
    "color1", 
    "color2",
    "color3"
]

X_bounds = [
    (0.,upper),
    (0.,upper),
    (0.,upper)
]

def get_space(normalize=False):
    space = {
        "color1":np.arange(0.,510.,10, dtype=np.float), 
        "color2":np.arange(0.,510.,10, dtype=np.float),
        "color3":np.arange(0.,510.,10, dtype=np.float)
    }
    
    if normalize:
        # normalize to [0,10]
        for i,k in enumerate(X_names):
            d = X_bounds[i][1] - X_bounds[i][0]
            if d > 0:
                space[k] = (space[k] - X_bounds[i][0]) / d
    return space

def get_domain(normalize=False):
    space = get_space(normalize)
    return [
    {"name":"color1", "type":'discrete', "domain":space["color1"]}, 
    {"name":"color2", "type":'discrete', "domain":space["color2"]}, 
    {"name":"color3", "type":'discrete', "domain":space["color3"]}
    ]

def read_table(filepath, normalize=False):
    init_design_df = pd.read_csv(filepath, index_col='id')
    init_design_arr = np.array([
        init_design_df["color1"].as_matrix(),
        init_design_df["color2"].as_matrix(),
        init_design_df["color3"].as_matrix()
    ], dtype=np.float).T
    
    init_result_arr = np.array([
        init_design_df["results"].as_matrix()
    ], dtype=np.float).T

    if normalize:
        # normalize to [0,1]
        for i,k in enumerate(X_names):
            d = X_bounds[i][1] - X_bounds[i][0]
            if d > 0:
                init_design_arr[:,i] = (init_design_arr[:,i] - X_bounds[i][0]) / d 
    return init_design_arr, init_result_arr

def rescale(X):
    one = np.ones(X.shape)
    init = np.ones(X.shape)
    for i,k in enumerate(X_names):
        d = X_bounds[i][1] - X_bounds[i][0]
        #print(d)
        one[:,i] = d
        init[:,i] = X_bounds[i][0]
    rescaledX = np.round(np.array(one*X + init),0)
    return rescaledX

def x2csv(nextX, savepath):
    result = nextX
    result = pd.DataFrame(result, columns=X_names)
    result.to_csv(savepath+"/csv/nextquery"+".csv")
    return 0


#simulator
def color_sim(x, target, minimize=True):
    def color(x):
        color1 = x[:,0]
        color2 = x[:,1]
        color3 = x[:,2]
        value450 = (color1*0.004 + color2*0.1405 + color3*0.273)/100
        value540 = (color1*0.0395 + color2*0.156 + color3*0.005)/100
        value620 = (color1*0.1345 + color2*0.012 + color3*0.0015)/100
        return np.array([value450, value540, value620])
    def calc_score(target_list, sample_list):
        sq = np.square(target_list - sample_list)
        score = np.sum(sq)
        return score
    
    x = np.array(x, dtype=np.float)
    target_x = np.array(target, dtype=np.float)

    result_x = color(x)
    result_target = color(target_x)
    y = calc_score(result_target, result_x)
    
    if minimize:
        return y
    else:
        return -y    

def color_sim_noisy(x, sd=0.001):
    return color_sim(x) + np.random.normal(loc=0, scale=sd)

def color(x):
    color1 = x[:,0]
    color2 = x[:,1]
    color3 = x[:,2]
    value450 = (color1*0.004 + color2*0.1405 + color3*0.273)/100
    value540 = (color1*0.0395 + color2*0.156 + color3*0.005)/100
    value620 = (color1*0.1345 + color2*0.012 + color3*0.0015)/100
    return np.array([value450, value540, value620])

def calc_score(target_list, sample_list):
    sq = np.square(target_list - sample_list)
    score = np.sum(sq)
    return score