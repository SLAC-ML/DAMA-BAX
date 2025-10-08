import os
import time
import re
from importlib import reload

import numpy as np
from tqdm.auto import trange
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.io import loadmat

import umap
from pyDOE import lhs
from pymoo.indicators.hv import HV

import da_NN as dann; reload(dann)
# DAMA-specific imports removed - should be imported by DAMA example, not core
# import da_ssrl as dass; reload(dass)
# import da_virtual_opt as davo; reload(davo)
# from dama_utils import evaluate_DA, evaluate_MA


def get_ith_gen_GT(i, data, n_pop=60):
    return data[i * n_pop:(i + 1) * n_pop]


def pareto_front(points):
    # Sort points by the first objective
    sorted_points = points[points[:, 0].argsort()]
    
    # Initialize the Pareto front list
    pareto_front = []
    
    # Track the best (lowest) second objective value encountered
    best_second_objective = float('inf')
    
    # Iterate through sorted points
    for point in sorted_points:
        if point[1] < best_second_objective:
            pareto_front.append(point)
            best_second_objective = point[1]
    
    return np.array(pareto_front)


def pareto_front_idx(points):
    # Sort points by the first objective
    idx_sorted = points[:, 0].argsort()
    sorted_points = points[idx_sorted]
    
    # Initialize the Pareto front list
    pareto_front = []
    idx_pf = []
    
    # Track the best (lowest) second objective value encountered
    best_second_objective = float('inf')
    
    # Iterate through sorted points
    for i, point in enumerate(sorted_points):
        if point[1] < best_second_objective:
            pareto_front.append(point)
            idx_pf.append(idx_sorted[i])
            best_second_objective = point[1]
    
    return np.array(pareto_front), np.array(idx_pf)


def plot_pf_evol(res):
    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='DA', ylabel='MA')

    for i in range(10):
        data = res.history[i].pop.get('F')
        ax.scatter(data[:, 0], data[:, 1], label=f'gen {i}')

    data = res.history[-1].pop.get('F')
    ax.scatter(data[:, 0], data[:, 1], color='k', label=f'final (gen {len(res.history) - 1})')

    # ax.grid()
    ax.legend(loc=0)
    fig.tight_layout()

    # fig.savefig('plot.pdf', bbox_inches='tight')
    plt.show()


def plot_pf_evol_GT(DAMA, n_pop=60, n_gen=93):
    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='DA', ylabel='MA')
    ax.set_title('Pareto front for each generation')

    for i in range(n_gen):
        data = get_ith_gen_GT(i, DAMA, n_pop=n_pop)
        pf = pareto_front(data)
        ax.scatter(pf[:, 0], pf[:, 1])
        
    pf = pareto_front(DAMA)
    ax.plot(pf[:, 0], pf[:, 1], '--', color='r', label=f'All generation combined')

    # data = get_ith_gen_GT(n_gen - 1, DAMA, n_pop=n_pop)
    # pf = pareto_front(data)
    # ax.scatter(pf[:, 0], pf[:, 1], color='k', label=f'final (gen {n_gen - 1})')

    # ax.grid()
    ax.legend(loc=0)
    fig.tight_layout()

    # fig.savefig('plot.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_pf_GT(DAMA):
    fig, ax = plt.subplots(1, 1)
    ax.set(xlabel='DA', ylabel='MA')
    ax.set_title('Pareto front for all generation combined')

    pf = pareto_front(DAMA)
    ax.scatter(pf[:, 0], pf[:, 1], color='r', label=f'Pareto front')

    # ax.grid()
    ax.legend(loc=0)
    fig.tight_layout()

    # fig.savefig('plot.pdf', bbox_inches='tight')
    plt.show()


def gen_plot_sol_dist_4d():
    # Load GT confs
    matdat = loadmat('data_DAMA4D_gen100.mat')

    SEXT_4D = matdat['g_dama'][:, 1:5]
    DAMA = matdat['g_dama'][:, 5:7]
    vrange4D = matdat['vrange4D']
    q = (SEXT_4D - vrange4D[:, 0]) / (vrange4D[:, 1] - vrange4D[:, 0])
    _, idx_pf = pareto_front_idx(DAMA)
    q_pf = q[idx_pf]

    reducer = umap.UMAP(random_state=42)
    reducer.fit(q)

    conf_gt_pf = reducer.transform(q_pf)
    conf_gt = reducer.transform(q)

    def plot_sol_dist(X, label=None):

        fig, ax = plt.subplots(1, 1)
        # ax.set(xlabel='', ylabel='')

        conf_inc = reducer.transform(X)

        if label is None:
            label = 'inc'

        ax.plot(conf_gt[:, 0], conf_gt[:, 1], 'o', color='grey', markersize=2, label='GT')
        ax.plot(conf_gt_pf[:, 0], conf_gt_pf[:, 1], 'o', color='b', markersize=4, label='GT Pareto front')
        ax.plot(conf_inc[:, 0], conf_inc[:, 1], 'o', color='r', markersize=4, label=label)

        # ax.grid()
        ax.legend(loc=0)
        fig.tight_layout()

        # fig.savefig('plot.pdf', bbox_inches='tight')
        plt.show()
        
    return plot_sol_dist


def norm_DAMA(DAMA):
    DAMA_n = (DAMA - DAMA.min(axis=0)) / (DAMA.max(axis=0) - DAMA.min(axis=0))
    
    return DAMA_n


def get_PF_region(DAMA, percentile=80, plot=False):
    
    DAMA_n = norm_DAMA(DAMA)
    PF_n = pareto_front(DAMA_n)

    n_conf = DAMA.shape[0]
    D_n = np.zeros(n_conf)

    for i in range(n_conf):
        D_n[i] = np.linalg.norm(PF_n - DAMA_n[i], axis=1).min()

    thres = -np.percentile(-D_n, percentile)
    indices_near = D_n <= thres
    DAMA_near = DAMA[indices_near]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set(xlabel='norm DA', ylabel='norm MA')
        ax.set_aspect('equal')
        ax.set_title(f'{percentile} percentile dist')

        DAMA_n_far = DAMA_n[D_n > thres]
        DAMA_n_near = DAMA_n[indices_near]

        ax.scatter(DAMA_n_far[:, 0], DAMA_n_far[:, 1], label=f'distance > {thres:.4f}', alpha=0.5, color='lightgrey')
        ax.scatter(DAMA_n_near[:, 0], DAMA_n_near[:, 1], label=f'distance <= {thres:.4f}', alpha=0.5)
        ax.plot(PF_n[:, 0], PF_n[:, 1], 'r--', alpha=1, label='Pareto front')

        # ax.grid()
        ax.legend(loc=0)
        fig.tight_layout()

        # fig.savefig('plot.pdf', bbox_inches='tight')
        plt.show()
        
    return DAMA_near, indices_near


# Pre-train utils
def convert_seconds_to_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hr{'s' if hours != 1 else ''}")
    if minutes > 0:
        time_parts.append(f"{minutes} min{'s' if minutes != 1 else ''}")
    if secs > 0 or not time_parts:
        if type(secs) is int:
            secs_str = f"{int(secs)} sec"
        else:
            secs_str = f"{secs:.1f} sec"
        secs_str += '' if secs == 1 else 's'
        time_parts.append(secs_str)

    return ' '.join(time_parts)


def gen_lhs_confs(n_conf=1):
    sext = lhs(4, samples=n_conf)

    return sext


def random_sel_confs(n_conf=1):
    idx_sel = np.random.choice(SEXT_unique.shape[0], size=n_conf, replace=False)
    sext = SEXT_unique[idx_sel, :]

    return sext


def gen_simulation_data(n_conf, avg_sample_per_conf, data_root, prefix, problem,
                        buffer_size=600, seed=42, target='DA'):
    
    np.random.seed(seed)
    
    n_pre = int(n_conf * avg_sample_per_conf)

    sext_confs = gen_lhs_confs(n_conf)

    assert sext_confs.shape[0] == n_conf
    # assert n_pre == 10000

    if target == 'DA':
        XX_confs, _ = problem.gen_X_data(sext_confs, exact=True)
    else:
        _, XX_confs = problem.gen_X_data(sext_confs)
    X_confs = dass.X_to_train_shape(XX_confs)

    
    selected_indices = np.random.choice(X_confs.shape[0], n_pre, replace=False)
    X_init = X_confs[selected_indices, :]

    assert X_init.shape[0] == n_pre
    seeds = np.random.choice(10, (X_init.shape[0], 1)) + 1  # seed range from 1 to 10
    _X_init = np.hstack([X_init, seeds])

    evaluate = evaluate_DA if (target == 'DA') else evaluate_MA
    Y_init = dass.get_Y_batch_sim_4d(_X_init, evaluate, data_root=data_root, prefix=prefix,
                                     buffer_size=buffer_size, format_data=False)
        
    return X_init, Y_init


def pretrain(X, Y, savefile, reloadfile=None, train_params=None, test_ratio=0.05,
             device=None, n_sel=None, seed=42,
             log=None, log_period=10, final_savefile=None,
             early_stop_patience=None, eval_mode_on_test=True, use_batch_loss=False):
    
    # Create the folder if needed
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    
    # Make copies since X/Y would be modified
    X = X.copy()
    Y = Y.copy()
    
    if n_sel is not None:
        if seed is not None:
            np.random.seed(seed)
            
        selected_indices = np.random.choice(X.shape[0], n_sel, replace=False)
        X = X[selected_indices, :]
        Y = Y[selected_indices]
        
        assert X.shape[0] == n_sel
        
    X_mu, X_std = dann.get_norm(X, eps=1e-5)
    X = dann.normalize(X, X_mu, X_std)
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    # create training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=1)
    
    # train params
    if train_params is None:
        epochs = 150
        train_noise = 0
        batch_size = 1000
        batch_size_test = batch_size
        dropout = 0.1
        n_neur = 800
        lr = 1e-4
        out_scale = 1
    else:
        epochs = train_params.get('epochs', 150)
        train_noise = train_params.get('train_noise', 0)
        batch_size = train_params.get('batch_size', 1000)
        batch_size_test = train_params.get('batch_size_test', batch_size)
        dropout = train_params.get('dropout', 0.1)
        n_neur = train_params.get('n_neur', 800)
        lr = train_params.get('lr', 1e-4)
        out_scale = train_params.get('out_scale', 1)

    # number of inputs is number of configuration parameters (q) plus x,y coordinates
    n_feat = 6

    trainset = dann.Dataset(X_train, Y_train)
    testset = dann.Dataset(X_test, Y_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                             shuffle=False, num_workers=1)

    # define network
    # "split" model type passes x, y coordinates again near the output
    danet = dann.DA_Net(dropout=dropout, train_noise=train_noise, n_feat=n_feat, n_neur=n_neur, device=device,
                        model_type='split', out_scale=out_scale).to(device)
    
    if reloadfile:
        danet.load_state_dict(torch.load(reloadfile, map_location=device))
        print(f'continue the training from {reloadfile}')

    t0 = time.time()
    danet = dann.train_NN_re(danet, trainloader, testloader, lr=lr, epochs=epochs,
                             savefile=savefile, final_savefile=final_savefile, device=device,
                             early_stop_patience=early_stop_patience,
                             log=log, log_period=log_period,
                             eval_mode_on_test=eval_mode_on_test, use_batch_loss=use_batch_loss)
    t1 = time.time()
    ts = convert_seconds_to_time(t1 - t0)
    print(f'pretrain time cost: {ts}')
    
    return danet


def load_pretrained_model(X_init, Y_init, reloadfile, train_params=None, device=None, n_sel=None, seed=42):
    
    if n_sel is not None:
        if seed is not None:
            np.random.seed(seed)
            
        selected_indices = np.random.choice(X_init.shape[0], n_sel, replace=False)
        X_init = X_init[selected_indices, :]
        Y_init = Y_init[selected_indices]
        
        assert X_init.shape[0] == n_sel
    
    X_mu, X_std = dann.get_norm(X_init, eps=1e-5)

    if train_params is None:
        train_noise = 0
        dropout = 0
        n_neur = 800
        out_scale = 1
    else:
        train_noise = train_params.get('train_noise', 0)
        dropout = train_params.get('dropout', 0)
        n_neur = train_params.get('n_neur', 800)
        out_scale = train_params.get('out_scale', 1)
    
    n_feat = 6
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    danet = dann.DA_Net(dropout=dropout, train_noise=train_noise, n_feat=n_feat, n_neur=n_neur, device=device,
                        model_type='split', out_scale=out_scale).to(device)

    danet.load_state_dict(torch.load(reloadfile, map_location=device))
    danet.eval()
    
    return danet, X_mu, X_std


# Algo utils
def gen_daR_mask(daR_idx, n_ext=5, npts_per_ray=72):
    daR_mask = np.zeros((daR_idx.shape[0], daR_idx.shape[1], npts_per_ray))

    for j in range(daR_idx.shape[0]):
        for i in range(daR_idx.shape[1]):
            idx = int(daR_idx[j, i])
            if n_ext > 0:
                daR_mask[j, i, idx - n_ext:idx + n_ext] = 1
            else:
                daR_mask[j, i, idx] = 1
        
    return daR_mask.astype(bool)


def gen_maPM_mask(maPM_idx, n_ext=3, npts_per_spos=49):
    maPM_mask = np.zeros((maPM_idx.shape[0], maPM_idx.shape[1], npts_per_spos))
    mid = npts_per_spos // 2

    for j in range(maPM_idx.shape[0]):
        for i in range(maPM_idx.shape[1]):
            idx_P = int(maPM_idx[j, i, 0])
            idx_M = int(maPM_idx[j, i, 1])
            
            if n_ext > 0:            
                lb_P = max(idx_P - n_ext, mid + 1)
                ub_P = min(idx_P + n_ext, npts_per_spos)
                lb_M = max(idx_M - n_ext, 0)
                ub_M = min(idx_M + n_ext, mid - 1)

                maPM_mask[j, i, lb_P:ub_P] = 1
                maPM_mask[j, i, lb_M:ub_M] = 1
            else:
                maPM_mask[j, i, idx_P] = 1
                maPM_mask[j, i, idx_M] = 1
        
    return maPM_mask.astype(bool)


def gen_mask_range(YY, lb=0.5, ub=0.95):
    assert lb <= ub, 'upper bound must be no less than lower bound!'
    mask = np.logical_and(YY >= lb, YY <= ub)
        
    return mask


def plot_pf_cmp(pf_list, pf_ref, label_list=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    ax.set(xlabel='DA', ylabel='MA')
    ax.set_title('Pareto front')

    for i in range(len(pf_list)):
        pf = pf_list[i]
        if label_list is not None:
            label = label_list[i]
        else:
            label = None
        ax.plot(pf[:, 0], pf[:, 1], '-o', label=label)
    
    ax.plot(pf_ref[:, 0], pf_ref[:, 1], '--o', color='r', label='GT')

    # ax.grid()
    ax.legend(loc=0)
    fig.tight_layout()

    # fig.savefig('plot.pdf', bbox_inches='tight')
    plt.show()
    

def plot_pf_region_cmp(pf_list, pf_ref, label_list=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    ax.set(xlabel='DA', ylabel='MA')
    ax.set_title('Pareto front region')

    for i in range(len(pf_list)):
        pf = pf_list[i]
        if label_list is not None:
            label = label_list[i]
        else:
            label = None
        ax.plot(pf[:, 0], pf[:, 1], 'o', label=label, alpha=0.5)
    
    ax.plot(pf_ref[:, 0], pf_ref[:, 1], 'o', color='r', label='GT', alpha=0.5)

    # ax.grid()
    ax.legend(loc=0)
    fig.tight_layout()

    # fig.savefig('plot.pdf', bbox_inches='tight')
    plt.show()


def get_pareto_set(info):
    pf_region_opt, pf_region_idx_opt, SEXT_opt_info, pf_region_GT_pred = info[:4]
    ps_region_opt = SEXT_opt_info[pf_region_idx_opt, :]
    pf_opt, pf_opt_idx = pareto_front_idx(pf_region_opt)
    ps_opt = ps_region_opt[pf_opt_idx]
    
    return ps_region_opt, ps_opt, pf_region_opt, pf_opt


def get_unique(X, Y=None):
    # X, Y are 2d array that have the same row number
    X_unique, X_idx_unique = np.unique(X, axis=0, return_index=True)
    
    if Y is None:
        return X_unique
    
    Y_unique = Y[X_idx_unique]
    
    return X_unique, Y_unique


def collect_ps_opt(pf_info):
    ps_opt_all = []
    for info in pf_info:
        ps_region_opt, ps_opt, pf_region_opt, pf_opt = get_pareto_set(info)
        ps_opt_all.append(ps_opt)
    ps_opt_all = np.vstack(ps_opt_all)
    
    # Get rid of all the duplicated solutions
    ps_opt_unique, ps_opt_idx_unique = np.unique(ps_opt_all, axis=0, return_index=True)
    print(f'collect ps opt - unique ratio: {ps_opt_unique.shape[0] / ps_opt_all.shape[0]:.2f}')
    
    return ps_opt_unique


def calc_hv(dama, ref_point=[-80, -1.7]):
    """
    Given the DAMA data (not necessarily a PF) and a reference point,
    calculate the hypervolume
    """
    pf = pareto_front(dama)
    hv = HV(ref_point=ref_point)
    
    return hv(pf)


def func_sel_npts(DAMA, SEXT, npts=10):
    pf, pf_idx = pareto_front_idx(DAMA)
    ps = SEXT[pf_idx]
    
    if pf.shape[0] <= npts:
        print(f'npts on PF ({pf.shape[0]}) <= npts to sample ({npts}), select full PF')
        return pf, ps
    
    indices = np.linspace(0, pf.shape[0] - 1, npts, dtype=int)  # Generate m evenly spaced indices
    DAMA_sel = pf[indices]
    SEXT_sel = ps[indices]
    
    assert DAMA_sel.shape[0] == npts
    
    return DAMA_sel, SEXT_sel


# BAX setup
def fn_factory(model, norm):
    """
    Create a surrogate function from a trained model.

    Parameters:
    -----------
    model : torch.nn.Module
        Trained neural network
    norm : function
        Normalization function

    Returns:
    --------
    fn : function
        Surrogate function X → Y_pred
    """
    def fn(X):
        # Normalize input
        X_n = norm(X)
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_n).to(model.device)
            y_pred = model(X_tensor).cpu().numpy()
        return y_pred.T  # (1, n) format

    return fn


class BAXOpt:

    def __init__(self, algo, fn_oracle, norm, init, device=None, snapshot=False, model_root='models', model_names=None):
        self.acquired_data = []
        self.model = []
        self.n_sampling = 200
        self.iter_idx = 0

        self.algo = algo
        self.fn_oracle = fn_oracle
        self.norm = norm
        self.init = init

        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device

        # If store all model weights in each loop
        # usually this is for dev/debug
        self.snapshot = snapshot
        self.model_root = model_root

        # Model naming (default: net0, net1, ...)
        n_obj = len(fn_oracle)
        if model_names is None:
            model_names = [f'net{i}' for i in range(n_obj)]
        elif len(model_names) != n_obj:
            raise ValueError(f"model_names must have length {n_obj} (number of objectives), got {len(model_names)}")
        self.model_names = model_names

        # Dataset params
        self.random_state = 1
        self.test_ratio = 0.05
        self.batch_size = 1000
        self.batch_size_test = self.batch_size
        self.weight_new_pts = 10

        # NN params
        self.n_feat = 6
        self.epochs = 150
        self.iter_epochs = 10
        self.train_noise = 0
        self.dropout = 0
        self.n_neur = 800
        self.lr = 1e-4
        self.out_scale = 1
        self.savefile = [
            os.path.join(model_root, f'{name}.pt')
            for name in model_names
        ]

    def update_acq_data(self, X, Y=None):
        # Note that X is in form [X0, X1, ...]
        # even for single obj, X is [X0]
        n_obj = len(self.fn_oracle)
        
        if Y is None:
            Y = []
            for i in range(n_obj):
                Yi = self.fn_oracle[i](X[i])
                
                if (len(Yi.shape) == 2) and (Yi.shape[1] == 1):
                    Yi = Yi.flatten()
                    
                Y.append(Yi)
        
        self.acquired_data.append((X, Y))  # raw X!
        
        if len(self.acquired_data) == 1:  # initialization
            self.X_train = []
            self.X_test = []
            self.Y_train = []
            self.Y_test = []
            self.weights_train = []
            self.trainset = []
            self.trainloader = []
            self.testset = []
            self.testloader = []
            
            for i in range(n_obj):
                Xi_n = self.norm[i](X[i])  # normalize the X data
                _X_train, _X_test, _Y_train, _Y_test = \
                    train_test_split(Xi_n, Y[i], test_size=self.test_ratio, random_state=self.random_state)
                _weights_train = np.ones(Y[i].shape)

                _trainset = dann.Dataset(_X_train, _Y_train)
                _trainloader = torch.utils.data.DataLoader(
                    _trainset, batch_size=self.batch_size, shuffle=True, num_workers=1)
                _testset = dann.Dataset(_X_test, _Y_test)
                _testloader = torch.utils.data.DataLoader(
                    _testset, batch_size=self.batch_size_test, shuffle=False, num_workers=1)
                
                self.X_train.append(_X_train)
                self.X_test.append(_X_test)
                self.Y_train.append(_Y_train)
                self.Y_test.append(_Y_test)
                self.weights_train.append(_weights_train)
                self.trainset.append(_trainset)
                self.trainloader.append(_trainloader)
                self.testset.append(_testset)
                self.testloader.append(_testloader)
        else:  # update train/test loader
            for i in range(n_obj):
                Xi_n = self.norm[i](X[i])  # normalize the X data
                X_batch_train, X_batch_test, Y_batch_train, Y_batch_test = \
                    train_test_split(Xi_n, Y[i], test_size=self.test_ratio, random_state=self.random_state)
                self.X_train[i] = np.concatenate((self.X_train[i], X_batch_train), axis=0)
                self.Y_train[i] = np.concatenate((self.Y_train[i], Y_batch_train), axis=0)
                weights_train_batch = self.weight_new_pts * np.ones(self.n_sampling)
                self.weights_train[i] = np.ones(self.weights_train[i].shape)  # reset the weight of the previous data 
                self.weights_train[i] = np.concatenate((self.weights_train[i], weights_train_batch), axis=0)
                self.X_test[i] = np.concatenate((self.X_test[i], X_batch_test), axis=0)
                self.Y_test[i] = np.concatenate((self.Y_test[i], Y_batch_test), axis=0)

                self.trainset[i] = dann.Dataset(self.X_train[i], self.Y_train[i], self.weights_train[i])
                self.trainloader[i] = torch.utils.data.DataLoader(
                    self.trainset[i], batch_size=self.batch_size, shuffle=True, num_workers=1)
                self.testset[i] = dann.Dataset(self.X_test[i], self.Y_test[i])
                self.testloader[i] = torch.utils.data.DataLoader(
                    self.testset[i], batch_size=self.batch_size_test, shuffle=False, num_workers=1)
    
    def update_posterior(self):
        if (len(self.acquired_data) == 1) and self.model:  # just initialized and model is provided
            print("Found pre-trained model, skip training")
            return  # skip the training since the model has already been trained
        
        n_obj = len(self.fn_oracle)
        
        t0 = time.time()
        
        if not self.model:
            for i in range(n_obj):
                net = dann.DA_Net(
                    dropout=self.dropout, train_noise=self.train_noise,
                    n_feat=self.n_feat, n_neur=self.n_neur, device=self.device,
                    model_type='split', out_scale=self.out_scale).to(self.device)
                self.model.append(net)
            epochs = self.epochs
        else:
            epochs = self.iter_epochs

        # Train the model
        for i in range(n_obj):
            if self.snapshot:  # save the model weights for every loop
                weight_name = f'{self.model_names[i]}_l{self.iter_idx}.pt'
                savefile = os.path.join(self.model_root, weight_name)
                weight_name_f = f'{self.model_names[i]}_l{self.iter_idx}_f.pt'
                final_savefile = os.path.join(self.model_root, weight_name_f)
            else:
                savefile = self.savefile[i]
                final_savefile = None
                
            # Create the folder if needed
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
            
            self.model[i] = dann.train_NN_re(
                self.model[i], self.trainloader[i], self.testloader[i], lr=self.lr, epochs=epochs,
                savefile=savefile, final_savefile=final_savefile, device=self.device,
                verbose=(len(self.acquired_data) == 1), early_stop_patience=10, log_period=10)
            
            # After training we should load the best model
            # self.model[i].load_state_dict(torch.load(savefile, map_location=self.device))
            
        t1 = time.time()
        ts = convert_seconds_to_time(t1 - t0)
        print(f'Model training time cost: {ts}')
        
    @property
    def fn(self):
        fn_list = []
        
        n_obj = len(self.fn_oracle)
        for i in range(n_obj):
            fn_list.append(fn_factory(self.model[i], self.norm[i]))
    
        return fn_list
    
    def identify_subspace(self):
        # self.X_sub is also a list!
        self.X_sub = self.algo(self.fn)
    
    def sampling(self):
        n_obj = len(self.fn_oracle)
        
        self.X_sampled = []
        for i in range(n_obj):
            n_sampling = np.min([self.n_sampling, self.X_sub[i].shape[0]])
            selected_indices = np.random.choice(self.X_sub[i].shape[0], n_sampling, replace=False)
            self.X_sampled.append(self.X_sub[i][selected_indices, :])
    
    def run_acquisition(self, n_iters=10, verbose=False):
        '''
        Outer loop for acquistion. 
        TODO: In principle should be general (given discrete data), since update_acq_data 
        can be arbitrarily complicated?
        '''
        
        for i_iter in range(n_iters) if verbose else trange(n_iters):
            if verbose:
                print(f"Running BAX iter {self.iter_idx}")
                
            n_obj = len(self.fn_oracle)
            
            t0 = time.time()
            
            if not self.iter_idx:  # first loop
                X0 = []
                Y0 = []
                for i in range(n_obj):
                    _X0, _Y0 = self.init[i]()
                    X0.append(_X0)
                    Y0.append(_Y0)
                    
                self.update_acq_data(X0, Y0)
                self.update_posterior()
                self.identify_subspace()
                self.sampling()
            else:
                self.update_acq_data(self.X_sampled)
                self.update_posterior()
                self.identify_subspace()
                self.sampling()
            
            t1 = time.time()
            ts = convert_seconds_to_time(t1 - t0)
            print(f'loop {self.iter_idx} time cost: {ts}')
                
            self.iter_idx += 1


def get_curr_loop_num(model_root, model_name='net1'):
    """
    Find the highest loop number from saved model checkpoints.

    Parameters:
    -----------
    model_root : str
        Directory containing model checkpoints
    model_name : str, optional
        Name of the model to search for (default: 'net1', the second objective)
        Looks for files matching pattern: {model_name}_l{N}_f.pt

    Returns:
    --------
    int or None
        Highest loop number found, or None if no checkpoints exist
    """
    if not os.path.isdir(model_root):
        return None  # Or raise an exception if preferred

    max_l = None
    # Pattern: {model_name}_l{N}_f.pt (e.g., net1_l5_f.pt, manet_l5_f.pt)
    pattern = re.compile(rf"{re.escape(model_name)}_l(\d+)_f\.pt")

    for fname in os.listdir(model_root):
        match = pattern.match(fname)
        if match:
            l_val = int(match.group(1))
            if max_l is None or l_val > max_l:
                max_l = l_val

    return max_l
