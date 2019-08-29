import os
import sys
import argparse
from generate_query import initial_design
from generate_query import calc_EIstep
from util import X_names, X_bounds
from util import read_table, x2csv


def get_args():
    parser = argparse.ArgumentParser()
    # params
    parser.add_argument("--inputpath", default="./input.csv")
    parser.add_argument("--savepath", default="./test")
    parser.add_argument("--batchsize", default=5)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--initdesign", default="latin") # "latin" or "random"
    parser.add_argument("--bokernel", default="RBF") # "matern52" or "RBF" or "Linear"
    parser.add_argument("--acquisition", default="EI") # "EI" or "UCB"
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    inputpath = args.inputpath
    savepath = args.savepath
    batchsize = args.batchsize
    normalize = args.normalize
    initdesign = args.initdesign
    kernel = args.bokernel
    acq = args.acquisition
    
    os.makedirs(savepath+"/model", exist_ok=True)
    os.makedirs(savepath+"/csv", exist_ok=True)
    
    X_init, Y_init = read_table(inputpath, normalize)
    
    if len(X_init)==0:
        nextX = initial_design(bounds=X_bounds, batchsize=batchsize, method=initdesign)
    else:
        nextX = calc_EIstep(X_init,Y_init,batchsize=batchsize,normalize =normalize, savepath= savepath,kernel=kernel)
    x2csv(nextX, savepath= savepath )
    
    
if __name__ == '__main__':
    main()