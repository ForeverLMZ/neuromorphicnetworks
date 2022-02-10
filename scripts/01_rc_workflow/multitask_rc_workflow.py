# -*- coding: utf-8 -*-
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

from conn2res import iodata
from conn2res import reservoir, coding
from conn2res import workflows

from netneurotools import networks
from netneurotools import datasets as d

from bct import reference

import copy
import random
import pandas as pd

#%% --------------------------------------------------------------------------------------------------------------------
# DYNAMIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------


N_PROCESS = 20#40
#N_RUNS = 1 #apple

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.path.exists(os.path.join(PROJ_DIR, 'data')):
        os.makedirs(os.path.join(PROJ_DIR, 'data'))
DATA_DIR = os.path.join(PROJ_DIR, 'data')

RAW_RES_DIR = os.path.join('/home/mingzeli/neuro/results/conn2res/', 'raw_results')

#%% --------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def run_workflow(connectome, method, ifNorm, task, FACTOR, path_io, path_res_sim, path_res_tsk, \
                 input_nodes, output_nodes, percentage = 0, rewireNum = 0, bin=False, readout_modules=None, \
                 scores_file=None, iter_id=None, iter_io=False, iter_sim=False, \
                 encode=True, decode=False, **kwargs):
    """
        Runs the full reservoir pipeline: loads and scales connectivity matrix,
        generates input/output data for the task, simulates reservoir states,
        and trains the readout module.

    """
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
        print(conn.shape)
    else:
        conn = np.random.rand(279,279)

    alphas = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5]

    if (task == "MemoryCapacity"):
        dataFrame= workflows.memory_capacity(conn, input_nodes, output_nodes, alphas = alphas, plot_res = False, input_gain = FACTOR, tau_max = 16)
        if ifNorm:
            dataFrame["score"] = (dataFrame["score"] / normalizeConn(connectome,conn))
        dataFrame.to_csv(os.path.join(path_res_tsk, encoding_file))
        print("saved as csv","  iter is ",iter_id)
        # except:
    else: #tasks are regular tasks from neurogym
        # scale weights [0,1]
        if bin: conn = conn.astype(bool).astype(int)
        else:   conn = (conn-conn.min())/(conn.max()-conn.min())
    
        # normalize by the spectral radius
        ew, _ = eigh(conn)
        conn  = conn/np.max(ew)
        n_nodes = len(conn)

        tasks = iodata.get_available_tasks() #??? returns the full list but maybe we don't want the full list
        for task in tasks[:]:
            x, y = iodata.fetch_dataset(task)
            n_samples  = x.shape[0]
            print(f'n_observations = {n_samples}')

            n_features = x.shape[1]
            print(f'n_features = {n_features}')

            try:    n_labels = y.shape[1] 
            except: n_labels = 1
            print(f'n_labels   = {n_labels}')

            fig, axs = plt.subplots(2,1, figsize=(10,10), sharex=True)
            axs = axs.ravel()
            axs[0].plot(x)
            axs[0].set_ylabel('Inputs')
            
            axs[1].plot(y)
            axs[1].set_ylabel('Outputs')

            plt.suptitle(task)
            plt.show()
            plt.close()

            # split data into training and test sets
            x_train, x_test = iodata.split_dataset(x)
            y_train, y_test = iodata.split_dataset(y)

            #we have already defined our input and output nodes

            # create input connectivity matrix, which defines the connec-
            # tions between the input layer (source nodes where the input signal is
            # coming from) and the input nodes of the reservoir.
            w_in = np.zeros((n_features, n_reservoir_nodes))
            w_in[np.ix_(np.arange(n_features), input_nodes)] = 1.0 # factor that modulates the activation state of the reservoir

            #evaluate network performance across various dynamical regimes
            for alpha in alphas[1:]:
            
                #print(f'\n----------------------- alpha = {alpha} -----------------------')

                # instantiate an Echo State Network object
                ESN = reservoir.EchoStateNetwork(w_ih=w_in,
                                                w_hh=alpha*conn.copy(),
                                                activation_function='tanh',
                                                )

                # simulate reservoir states; select only output nodes.
                rs_train = ESN.simulate(ext_input=x_train)[:,output_nodes]
                rs_test  = ESN.simulate(ext_input=x_test)[:,output_nodes] 

                # perform task
                df = coding.encoder(reservoir_states=(rs_train, rs_test),
                                    target=(y_train, y_test),
                                    readout_modules=rsn_mapping,
                                    # pttn_lens=()
                                    )

                df['alpha'] = np.round(alpha, 3)

                # reorganize the columns
                if 'module' in df.columns:
                    df_subj.append(df[['module', 'n_nodes', 'alpha', 'score']])
                else:
                    df_subj.append(df[['alpha', 'score']])
                
            df_subj = pd.concat(df_subj, ignore_index=True)
            df_subj['score'] = df_subj['score'].astype(float)
            df_subj['alpha'] = df_subj['alpha'].astype(float)

            #print(df_subj.head(5))
            
            #############################################################################
            # Now we plot the performance curve
            sns.set(style="ticks", font_scale=2.0)  
            fig = plt.figure(num=1, figsize=(12,10))
            ax = plt.subplot(111)
            sns.lineplot(data=df_subj, x='alpha', y='score', 
                        hue='module', 
                        # hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
                        palette=sns.color_palette('husl', 7), 
                        markers=True, 
                        ax=ax)
            sns.despine(offset=10, trim=True)
            plt.title(task)
            plt.plot()
            plt.show()
            

#%% --------------------------------------------------------------------------------------------------------------------
# LOCAL
# ----------------------------------------------------------------------------------------------------------------------
def reliability(connectome, FACTOR, ifNorm, method, task):
    if (method == 'reverse'):
        N_RUNS = 1
    else:
        N_RUNS = 1000#change back

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
    #data = d.fetch_connectome(connectome) #apple
    # data = dict()
    # data['conn'] = np.load("/home/mingzeli/neuro/neuromorphicnetworks/data/drosophila_conn.npy")
    # #np.save(os.path.join(DATA_DIR, connectome + '_conn'+'.npy'), data['conn'])
    # if ifNorm:
    #     #np.save(os.path.join(DATA_DIR, connectome + '_dist'+'.npy'), data['dist']) #apple
    #     data['dist'] = np.load("/home/mingzeli/neuro/neuromorphicnetworks/data/drosophila_dist.npy")
    
    # --------------------------------------------------------------------------------------------------------------------
    # RUN WORKFLOW
    # ----------------------------------------------------------------------------------------------------------------------
    params = []
    #randomNums=np.random.randint(1000, size=(N_RUNS))
    for run in range(N_RUNS):
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
            output_nodes_temp = [7,8,9,15,19,33,39] #premotor
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
        elif(connectome == 'macaque_markov'):
            output_nodes_temp = [8] #motor
            input_nodes_temp = [0,1,2,3] #visual
        elif(connectome == 'marmoset'):
            input_nodes_temp = [48,49,50,51,52,53,54]
            output_nodes_temp = [16,17,18,19,20,21]
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
               'iter_id':run,
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
               'task': task,
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
    connectomes = ['macaque_markov']
    #factors = [0.001]#, 0.01, 0.1, 1, 10]
    factor = 1.0
    normalizations = [True,False]
    methods = ['empirical','rewire','reverse']
    task = 'MemoryCapacity'

    for connectome in connectomes:
        for normalization in normalizations:
                for method in methods:
                    reliability(connectome, factor, normalization, method, task)
                    print ("finished with ", str(connectome), str(factor),str(normalization), str(method))

if __name__ == '__main__':
    main()
