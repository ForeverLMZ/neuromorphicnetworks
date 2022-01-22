# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:27:07 2020

@author: Estefany Suarez
"""


import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

import scipy.io as sio
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

from reservoir.network import nulls
from reservoir.tasks import (io, coding, tasks)
from reservoir.simulator import sim_lnm

from netneurotools import networks
from netneurotools import datasets as d

from bct import reference

import copy
import random

#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
TASK = 'sgnl_recon' #'sgnl_recon' 'pttn_recog'
SPEC_TASK = 'mem_cap' #'mem_cap' 'nonlin_cap' 'fcn_app'
TASK_REF = 'T1'
#FACTOR = 0.0001 #0.0001 0.001 0.01

#INPUTS = 'subctx'
CLASS = 'functional' #'functional' 'cytoarch' #only used in metadata, which only used in null and spintest

N_PROCESS = 20#40
#N_RUNS = 1 #apple

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.path.exists(os.path.join(PROJ_DIR, 'data')):
        os.makedirs(os.path.join(PROJ_DIR, 'data'))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def load_metada(connectome, include_subctx=False):

    ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))

    if CLASS == 'functional':
        filename = CLASS
        class_labels = np.array(['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'])
        class_mapping = np.load(os.path.join(DATA_DIR, 'rsn_mapping', 'rsn_' + connectome + '.npy'))
        class_mapping_ctx = class_mapping[ctx == 1]

    elif CLASS == 'cytoarch':
        filename = CLASS
        class_labels = np.array(['PM', 'AC1', 'AC2', 'PSS', 'PS', 'LIM', 'IC'])
        class_mapping = np.load(os.path.join(DATA_DIR, 'cyto_mapping', 'cyto_' + connectome + '.npy'))
        class_mapping_ctx = class_mapping[ctx == 1]

    if include_subctx:
        return filename, class_labels, class_mapping

    else:
        return filename, class_labels, class_mapping_ctx


def consensus_network(connectome, coords, hemiid, path_res_conn, iter_id=None, sample=None, **kwargs):

    if iter_id is not None: conn_file = f'consensus_{iter_id}.npy'

    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        # load connectivity data
        CONN_DIR = os.path.join(DATA_DIR, 'connectivity', 'individual')
        stru_conn = np.load(os.path.join(CONN_DIR, connectome + '.npy'))

        # remove subctx
        # ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))
        # idx_ctx, = np.where(ctx == 1)
        # stru_conn = stru_conn[np.ix_(idx_ctx, idx_ctx)]

        # remove nans
        nan_subjs = np.unique(np.where(np.isnan(stru_conn))[-1])
        stru_conn = np.delete(stru_conn, nan_subjs, axis=2)
        stru_conn_avg = networks.struct_consensus(data=stru_conn.copy()[:,:,sample],
                                                  distance=cdist(coords, coords, metric='euclidean'),
                                                  hemiid=hemiid[:, np.newaxis]
                                                  )

        stru_conn_avg = stru_conn_avg*np.mean(stru_conn, axis=2)

        np.save(os.path.join(path_res_conn, conn_file), stru_conn_avg)


def null_network(model_name, path_res_conn, iter_id=None, **kwargs):

    if iter_id is not None: conn_file = f'{model_name}_{iter_id}' + '.npy'

    if not os.path.exists(os.path.join(path_res_conn, conn_file)):

        new_conn = nulls.construct_null_model(type=model_name, **kwargs)

        np.save(os.path.join(path_res_conn, conn_file), new_conn)


def run_workflow(connectome, method, ifNorm, FACTOR, path_io, path_res_sim, path_res_tsk, \
                 input_nodes, output_nodes, percentage = 0,rewireNum = 0, bin=False, readout_modules=None, \
                 scores_file=None, iter_id=None, iter_io=False, iter_sim=False, \
                 encode=True, decode=False, **kwargs):
    """
        Runs the full reservoir pipeline: loads and scales connectivity matrix,
        generates input/output data for the task, simulates reservoir states,
        and trains the readout module.

        Parameters
        ----------
        conn_name: str, {'consensus', 'rand_mio'}
            Specifies the name of the connectivity matrix file. 'consensus' for
            reliability and spintest analyses, and 'rand_mio' for significance
            analysis. #omitted

        connectome: str, {human_500, human_250}
            Specifies the scale of the conenctome

        method: str,{rewire, reverse}
            specifies the type of null network we are dealing with

        path_res_conn : str
            Path to conenctivity matrix

        path_io : str
            Path to simulation results

        path_res_sim : str
            Path to simulation results

        path_res_tsk : str
            Path to task scores

        bin : bool
            If True, the binary matrix will be used

        input,output nodes: (N,) list or numpy.darray
            List or array that indicates the indexes of the input and output
            nodes in the recurrent network.
            N: number of input,output nodes in the network

        readout_modules: (N,) numpy.darray, optional
            Array that indicates the module at which each output node belongs
            to. Modules can be int or str.
            N: number of output nodes

        scores_file : str, {'functional', 'cytoarch'}
            Name of the partition used

        iter_id : int
            Number/name of the iteration

        iter_{conn,io,sim} : bool
            If True, specific instances (i.e., connectivity, input/output data,
            network states) related to the iteration indicated by iter_id will
            be used.

        encode,decode : bool
            If True, encoding,decoding will run
    """

    # --------------------------------------------------------------------------------------------------------------------
    # DEFINE FILE NAMES
    # ----------------------------------------------------------------------------------------------------------------------

    # define file I/O data
    if np.logical_and(iter_id is not None, iter_io):
        input_file  = 'inputs_' + str(iter_id) + '.npy'
        output_file = 'outputs_' + str(iter_id) + '.npy'
    else:
        input_file  = 'inputs.npy'
        output_file = 'outputs.npy'

    # define file simulation data (reservoir states)
    if np.logical_and(iter_id is not None, iter_sim):
        res_states_file = 'reservoir_states_' + str(iter_id) + '.npy'
    else: res_states_file  = 'reservoir_states.npy'

    # define file encoding/decoding scores data
    if np.logical_and(iter_id is not None, scores_file is not None):
        encoding_file = scores_file + '_encoding_score_' + str(iter_id) + '.csv'
        decoding_file = scores_file + '_decoding_score_' + str(iter_id) + '.csv'

    elif np.logical_and(iter_id is not None, scores_file is None):
        encoding_file = 'encoding_score_' + str(iter_id) + '.csv'
        decoding_file = 'decoding_score_' + str(iter_id) + '.csv'

    elif np.logical_and(iter_id is None, scores_file is not None):
        encoding_file = scores_file + '_encoding_score.csv'
        decoding_file = scores_file + '_decoding_score.csv'

    else:
        encoding_file = 'encoding_score.csv'
        decoding_file = 'decoding_score.csv'

    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT CONNECTIVITY DATA
    # ----------------------------------------------------------------------------------------------------------------------
    if (method == 'reverse'):
        # load connectivity data
        conn = reversedConn(connectome,percentage)
    elif(method == 'rewire'):
        if (method != 'rewire' or rewireNum == 0):
            print("this script is not working right!!")
        conn,num  = rewiringConn(connectome,rewireNum)
    elif(method == 'empirical'):
        conn = np.load(os.path.join(DATA_DIR, connectome + '_conn'+'.npy'), allow_pickle=True)
    else:
        conn = np.random.rand(279,279)
        
    
    #ctx = np.load(os.path.join(DATA_DIR, 'cortical', 'cortical_' + connectome + '.npy'))

    # scale weights [0,1]
    if bin: conn = conn.astype(bool).astype(int)
    else:   conn = (conn-conn.min())/(conn.max()-conn.min())
 
    # normalize by the spectral radius
    ew, _ = eigh(conn)
    conn  = conn/np.max(ew)
    n_nodes = len(conn)

    # select input nodes
    #if input_nodes is None: input_nodes = np.where(ctx == 0)[0] #subcortical
    #if output_nodes is None: output_nodes = np.where(ctx == 1)[0] #cortical

    # --------------------------------------------------------------------------------------------------------------------
    # CREATE I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(os.path.join(path_io, input_file)):

        io_kwargs = {'time_len':2050}

        inputs, outputs = io.get_io_data(task=TASK,
                                         task_ref=TASK_REF,
                                         **io_kwargs
                                        )

        np.save(os.path.join(path_io, input_file), inputs)
        np.save(os.path.join(path_io, output_file), outputs)


    # --------------------------------------------------------------------------------------------------------------------
    # NETWORK SIMULATION - LINEAR MODEL
    # ----------------------------------------------------------------------------------------------------------------------
    alphas = tasks.get_default_alpha_values(SPEC_TASK)
    if not os.path.exists(os.path.join(path_res_sim, res_states_file)):

        input_train, input_test = np.load(os.path.join(path_io, input_file))

        # create input connectivity matrix - depends on the shape of the input
        w_in = np.zeros((input_train.shape[1],len(conn)))
        w_in[:,input_nodes] = FACTOR

        reservoir_states_train = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_train,
                                                 alphas=alphas,
                                                )

        reservoir_states_test  = sim_lnm.run_sim(w_in=w_in,
                                                 w=conn,
                                                 inputs=input_test,
                                                 alphas=alphas,
                                                )

        reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(reservoir_states_train, reservoir_states_test)]
        np.save(os.path.join(path_res_sim, res_states_file), reservoir_states, allow_pickle=False)


    # --------------------------------------------------------------------------------------------------------------------
    # IMPORT I/O DATA FOR TASK
    # ----------------------------------------------------------------------------------------------------------------------
    reservoir_states = np.load(os.path.join(path_res_sim, res_states_file), allow_pickle=True)
    reservoir_states = reservoir_states[:, :, :, output_nodes]
    reservoir_states = reservoir_states.squeeze()
    reservoir_states = np.split(reservoir_states, len(reservoir_states), axis=0)
    reservoir_states = [rs.squeeze() for rs in reservoir_states]
    

    outputs = np.load(os.path.join(path_io, output_file))

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - ENCODERS
    # ----------------------------------------------------------------------------------------------------------------------
    #print("started encoding")
    a= encode
    b = not os.path.exists(os.path.join(path_res_tsk, encoding_file))
    #print(a,b)
    if np.logical_and(a, b):
        
        #print('\nEncoding: ') #prints this
        df_encoding = coding.encoder(task=SPEC_TASK,
                                    target=outputs,
                                    reservoir_states=reservoir_states,
                                    readout_modules=None,
                                    alphas=alphas,
                                    )

        # df_encoding = df_encoding.rename(columns={'module':'class'}, copy=False)
        if ifNorm:
            df_encoding["performance"] = (df_encoding["performance"] / normalizeConn(connectome,conn)) #apple #normalization   
        df_encoding.to_csv(os.path.join(path_res_tsk, encoding_file))
        print("saved as csv","  iter is ",iter_id)
        # except:
        #     print("went to pass")
        # #    print(a,b)
        # #     pass

    # --------------------------------------------------------------------------------------------------------------------
    # PERFORM TASK - DECODERS
    # ----------------------------------------------------------------------------------------------------------------------
    try:
        if np.logical_and(decode, not os.path.exists(os.path.join(path_res_tsk, decoding_file))):

            # binarize cortical adjacency matrix
            conn_bin = conn.copy()[np.ix_(np.where(ctx==1)[0], np.where(ctx==1)[0])].astype(bool).astype(int)

            print('\nDecoding: ')
            df_decoding = coding.decoder(task=SPEC_TASK,
                                         target=outputs,
                                         reservoir_states=reservoir_states,
                                         readout_modules=readout_modules,
                                         bin_conn=conn_bin,
                                         alphas=alphas,
                                         )

            df_decoding = df_decoding.rename(columns={'module':'class'}, copy=False)
            df_decoding.to_csv(os.path.join(path_res_tsk, decoding_file))

    except:
        pass

    # delete reservoir states to release memory storage
    if iter_sim: os.remove(os.path.join(path_res_sim, res_states_file)) #apple


#%% --------------------------------------------------------------------------------------------------------------------
# LOCAL
# ----------------------------------------------------------------------------------------------------------------------
def reliability(connectome, FACTOR, ifNorm, method):
    if (method == 'rewire'):
        N_RUNS = 1000
    else:
        N_RUNS = 1

    print ('INITIATING PROCESSING TIME - RELIABILITY')
    t0_1 = time.perf_counter()
    #t0_2 = time.time()


    EXP = 'reliability'
    IO_TASK_DIR  = os.path.join(RAW_RES_DIR, 'io_tasks', EXP)
    RES_CONN_DIR = os.path.join(RAW_RES_DIR, 'conn_results', EXP)
    RES_SIM_DIR  = os.path.join(RAW_RES_DIR, 'sim_results', EXP)# f'{INPUTS}_scale{connectome[-3:]}'
    RES_TSK_DIR  = os.path.join(RAW_RES_DIR, 'tsk_results', EXP)

    if not os.path.exists(IO_TASK_DIR):  os.makedirs(IO_TASK_DIR)
    if not os.path.exists(RES_CONN_DIR): os.makedirs(RES_CONN_DIR)
    if not os.path.exists(RES_SIM_DIR):  os.makedirs(RES_SIM_DIR)
    if not os.path.exists(RES_TSK_DIR):  os.makedirs(RES_TSK_DIR)

    # define file connectivity data
    data = d.fetch_connectome(connectome)
    np.save(os.path.join(DATA_DIR, connectome + '_conn'+'.npy'), data['conn'])
    if ifNorm:
        np.save(os.path.join(DATA_DIR, connectome + '_dist'+'.npy'), data['dist'])
    
    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    params = []
  
    for node in range(N_RUNS):
        if (connectome == 'celegans'):
            input_nodes_temp = [2,3,139,140,141,142,143,144] #sensory nodes
                    # #motor neurons only for celegans
            output_nodes_temp = [70,109,113,114,135,136,155,156,177,178,199,200]
            output_nodes_temp.extend(range(28,39))
            output_nodes_temp.extend(range(86,108))
            output_nodes_temp.extend(range(185,197))
            output_nodes_temp.extend(range(218,230))
            output_nodes_temp.extend(range(238,279))
        elif (connectome == 'drosophila'):
            input_nodes_temp = [11, 12, 14, 16, 17, 23, 24, 35, 36, 38, 40, 41, 46,47] #vis
            # #all other neurons as output neurons
            output_nodes_temp = list(range(len(data['conn'])))
            output_nodes_temp = np.delete(output_nodes_temp, input_nodes_temp)
        elif(connectome == 'macaque_modha'):
            output_nodes_temp = [55,56,57] 
            output_nodes_temp.extend(range(59,76)) #motor module
            input_nodes_temp = range(221,242)
        elif(connectome == 'mouse'):
            output_nodes_temp = [84,85] #motor areas
            input_nodes_temp = [66,10,11,12,13,14] #visual areas
            #input_nodes_temp = [4,5,15,19,60,61,71,75] #olfac areas
        elif(connectome == 'human_func_scale250'):
            output_nodes_temp = []
            output_nodes_temp.extend(range(58,74))
            output_nodes_temp.extend(range(288,304)) # precentral
            input_nodes_temp = []
            input_nodes_temp.extend(range (147, 165))
            input_nodes_temp.extend(range(379,396)) # cuneus, pericalcarine and laeraloccipital

        else:
            input_nodes_temp = [42]
        # add output nodes



        #method = 'empirical'#apple
        rewireNum = 10#apple
        percentage  = 1
        tmp = {'connectome':connectome,
                'percentage':percentage, #only matters when method is reverse #apple
               'scores_file':(str(method) + "_"+ str(FACTOR)+ "_"+ str(connectome)+"_"+ str(ifNorm)), #apple # str(rewireNum)
               'rewireNum': rewireNum, #only matters when method is rewire
               'method':method, #{reverse,rewire}
               'ifNorm':ifNorm,
               'FACTOR':FACTOR,
               'iter_id':node,
               'iter_conn':True,
               'iter_io':False,
               'iter_sim':True,
               'encode':True,
               'decode':False,
               'readout_modules':None,
               'input_nodes':input_nodes_temp, 
               'output_nodes':output_nodes_temp,
               'path_res_conn':RES_CONN_DIR,
               'path_io':IO_TASK_DIR,
               'path_res_sim':RES_SIM_DIR,
               'path_res_tsk':RES_TSK_DIR,
                }

        params.append(tmp)

    pool2 = mp.Pool(processes=N_PROCESS)
    res2 = [pool2.apply_async(run_workflow, (), p) for p in params]
    for r2 in res2: r2.get()
    pool2.close()

    print ('PROCESSING TIME - RELIABILITY')
    print (time.perf_counter()-t0_1, "seconds process time")
    #print (time.time()-t0_2, "seconds wall time")

def reversedConn (connectome,percentage):#to be called after data is fetched in reliability
    conn = np.load(os.path.join(DATA_DIR, connectome + '_conn'+'.npy'), allow_pickle=True)
    if percentage == 0:
        return conn
    else:
        if percentage == 1:
            reverseAll = True
            rConn = conn.transpose()
            return rConn
        else:
            reverseAll = False
            sampledList = selectPairs(connectome,percentage) #the list contain indices pairs that are to be reversed
            #swaps
            rConn = copy.deepcopy(conn)
            for pair in sampledList:
                temp = rConn[pair[0]][pair[1]]
                rConn[pair[0]][pair[1]] = rConn[pair[1]][pair[0]]
                rConn[pair[1]][pair[0]] = temp

        return rConn

def rewiringConn(connectome,rewireIter):
    conn = np.load(os.path.join(DATA_DIR, connectome + '_conn'+'.npy'), allow_pickle=True)
    arr,num = reference.randmio_dir(conn, rewireIter)
    return arr,num

def normalizeConn(connectome,conn):
    dist = np.load(os.path.join(DATA_DIR, connectome + '_dist'+'.npy'), allow_pickle=True)
    wiringCost = np.sum(dist*conn)
    return wiringCost


#%% --------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------
def main():
    connectomes = ['human_func_scale250']#, 'celegans','mouse']
    factors = [0.001]#, 0.01, 0.1, 1, 10]
    normalizations = [False]
    methods = ['empirical', 'rewire','reverse']

    for connectome in connectomes:
        for factor in factors:
            for normalization in normalizations:
                    for method in methods:
                        reliability(connectome, factor, normalization, method)
                        print ("finished with ", str(connectome), str(factor),str(normalization), str(method))

if __name__ == '__main__':
    main()
