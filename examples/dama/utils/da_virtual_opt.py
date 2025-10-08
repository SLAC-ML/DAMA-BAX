from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import scipy.io as scio
from scipy.interpolate import griddata
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Add core to path for da_NN
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../core'))
import da_NN as dann

from . import da_ssrl as dass
from .da_utils import gen_ray_angles, gen_ray_grid_xy, gen_ray_grid_ar
from .da_utils import gen_spos, gen_momentum, gen_grid_sm


def predict_turns(X, danet, X_mu, X_std, max_pts=100000, verbose=1):
    # X: train-shape X before normalization
    
    _X = X.copy()
    _X = dann.normalize(_X, X_mu, X_std)

    a = X[:, -2]
    r = X[:, -1]
    
    device = next(danet.parameters()).device
    p = dann.predict_turns(_X, danet=danet, nsamp=1, max_pts=max_pts, device=device, verbose=verbose)
    
    return a, r, p


def predict_turns_alt(X, danet, norm, max_pts=100000, verbose=1):
    # X: train-shape X before normalization
    
    _X = X.copy()
    _X = norm(_X)

    a = X[:, -2]  # spos (int) for MA
    r = X[:, -1]  # momentum for MA
    
    device = next(danet.parameters()).device
    p = dann.predict_turns(_X, danet=danet, nsamp=1, max_pts=max_pts, device=device, verbose=verbose)
    
    return a, r, p


def get_top_solutions(res, top_percentage=0.1):
    variables = res.pop.get("X")
    objectives = res.pop.get("F")

    # Sort by objective value
    sorted_indices = np.argsort(objectives[:, 0])  # Sort by the first column

    # Extract the sorted variables and objectives
    sorted_variables = variables[sorted_indices]
    sorted_objectives = objectives[sorted_indices]

    # Define the top X% (e.g., top 10%)
    top_count = int(len(sorted_variables) * top_percentage)

    # Get the top solutions (variables and objectives)
    top_variables = sorted_variables[:top_count]
    top_objectives = sorted_objectives[:top_count]

    return top_objectives, top_variables


def sel_random_solutions(res, size=100):
    variables = res.pop.get("X")
    objectives = res.pop.get("F")

    # Sort by objective value
    n_tot = objectives.shape[0]
    n_sel = min(size, n_tot)
    indices_sel = np.random.choice(np.arange(n_tot), size=n_sel, replace=False)

    # Extract the sorted variables and objectives
    variables_sel = variables[indices_sel]
    objectives_sel = objectives[indices_sel]

    return objectives_sel, variables_sel, res


class VirtualDA(Problem):

    def __init__(self, danet=None, method='xiaobiao',
                 da_thresh=0.95, da_method=0, verbose=1):
        
        lb_sext, ub_sext = dass.get_srange()
        super().__init__(n_var=6,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=lb_sext,
                         xu=ub_sext)
        
        self._danet = danet
        self._method = method
        self._da_thresh = da_thresh
        self._da_method = da_method
        self._verbose = verbose
        
        # Load the aarr part of Daniel's method
        npts_per_ray = 29
        dat = dass.load_dat(mypath='')
        daX = dat['daX']
        daY = dat['daY']
        q = dat['pf']
        daR = np.sqrt(daX**2 + daY**2)
        self.angles_daniel = angles = dass.gen_ray_angles(daX, daY, samp=0)
        aa = np.repeat(angles[:, np.newaxis], npts_per_ray, 1)
        XX_infer, rr = dass.gen_X_data(daR, q, angles, npts_per_ray, type='inference')
        self.rr_daniel = rr
        self.aarr_daniel = np.concatenate((aa[:, :, np.newaxis], rr[:, :, np.newaxis]), axis=2)
        
        X_infer = dass.X_to_train_shape(XX_infer)
        self.X_mu_daniel, self.X_std_daniel = dann.get_norm(X_infer, eps=1e-5)
        
        # Calculate aarr part of Xiaobiao's method
        npts_per_ray = 72
        self.angles_xiaobiao = self.angles_daniel  # angles are the same
        
        xx, yy = gen_ray_grid_xy(npts_per_ray=npts_per_ray)
        self.xxyy_xiaobiao = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]), axis=2)
        aa, rr = gen_ray_grid_ar(npts_per_ray=npts_per_ray)
        self.rr_xiaobiao = rr
        self.aarr_xiaobiao = np.concatenate((aa[:, :, np.newaxis], rr[:, :, np.newaxis]), axis=2)
        
        XX_infer_xiaobiao, _ = dass.gen_X_data(daR, q, self.angles_xiaobiao, npts_per_ray, type='xiaobiao')
        X_infer_xiaobiao = dass.X_to_train_shape(XX_infer_xiaobiao)
        self.X_mu_xiaobiao, self.X_std_xiaobiao = dann.get_norm(X_infer_xiaobiao, eps=1e-5)
        
        if method is 'xiaobiao':
            self._rr = self.rr_xiaobiao
            self._angles = self.angles_xiaobiao
            self.X_mu = self.X_mu_xiaobiao
            self.X_std = self.X_std_xiaobiao
        elif method is 'daniel':
            self._rr = self.rr_daniel
            self._angles = self.angles_daniel
            self.X_mu = self.X_mu_daniel
            self.X_std = self.X_std_daniel
        else:
            raise NotImplementedError
            
    def set_danet(self, danet):
        self._danet = danet

    def gen_X_data(self, q, exact=False):
        # q: (n, 6)
        
        nq, _ = q.shape

        if self._method is 'xiaobiao':
            nrays = 19
            npts_per_ray = 72

            qq = np.repeat(q[:, np.newaxis, :], nrays, 1)
            qq2 = np.repeat(qq[:, :, np.newaxis, :], npts_per_ray, 2)

            if exact:
                xxyy2 = np.repeat(self.xxyy_xiaobiao[np.newaxis, :, :], nq, 0)

                XX = np.concatenate([qq2, xxyy2], axis=-1)
            else:
                aarr2 = np.repeat(self.aarr_xiaobiao[np.newaxis, :, :], nq, 0)

                XX = np.concatenate([qq2, aarr2], axis=-1)
        elif self._method is 'daniel':
            nrays = 19
            npts_per_ray = 29

            qq = np.repeat(q[:, np.newaxis, :], nrays, 1)
            qq2 = np.repeat(qq[:, :, np.newaxis, :], npts_per_ray, 2)
            
            aarr2 = np.repeat(self.aarr_daniel[np.newaxis, :, :], nq, 0)
            
            XX = np.concatenate([qq2, aarr2], axis=-1)
        else:
            raise NotImplementedError

        return XX
    
    def get_obj_scaled(self, x):
        # x: (n, 6)
        
        return None  # ignore obj_scaled for now (obj_scaled = 1.0)
        
        n, _ = x.shape
        
        s0 = np.array([[0.8578286, -162.44251681]])
        obj_scaled = np.repeat(s0, n, 0)
        
        return obj_scaled


    def _evaluate(self, x, out, *args, **kwargs):
        # x: (n, 6)
        
        XX = self.gen_X_data(x)
        X = dass.X_to_train_shape(XX)
        _, _, y_pred = predict_turns(X, self._danet, self.X_mu, self.X_std, verbose=self._verbose)

        obj_scaled = self.get_obj_scaled(x)
        obj_pred, _ = dass.pred_to_obj(
            y_pred, self._rr, self._angles, obj_scaled,
            da_thresh=self._da_thresh, method=self._da_method)

        out["F"] = obj_pred
        
        
class VirtualDAMA(Problem):

    def __init__(self, fn_da=None, fn_ma=None, method='xiaobiao',
                 da_thresh=0.95, da_method=0, ma_thresh=0.95, ma_method=1,
                 verbose=1, exact=False):
        
        lb_sext, ub_sext = np.zeros(4), np.ones(4)
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=lb_sext,
                         xu=ub_sext)
        
        self._fn_da = fn_da
        self._fn_ma = fn_ma
        self._method = method
        self._da_thresh = da_thresh
        self._da_method = da_method
        self._ma_thresh = ma_thresh
        self._ma_method = ma_method
        self._verbose = verbose
        self._exact = exact
        
        # Load the aarr part of Daniel's method
        npts_per_ray = 29
        dat = dass.load_dat(mypath='')
        daX = dat['daX']
        daY = dat['daY']
        q = dat['pf']
        daR = np.sqrt(daX**2 + daY**2)
        self.angles_daniel = angles = dass.gen_ray_angles(daX, daY, samp=0)
        aa = np.repeat(angles[:, np.newaxis], npts_per_ray, 1)
        XX_infer, rr = dass.gen_X_data(daR, q, angles, npts_per_ray, type='inference')
        self.rr_daniel = rr
        self.aarr_daniel = np.concatenate((aa[:, :, np.newaxis], rr[:, :, np.newaxis]), axis=2)
        
        X_infer = dass.X_to_train_shape(XX_infer)
        self.X_mu_daniel, self.X_std_daniel = dann.get_norm(X_infer, eps=1e-5)
        
        # Calculate aarr part of Xiaobiao's method
        npts_per_ray = 72
        self.angles_xiaobiao = self.angles_daniel  # angles are the same
        xx, yy = gen_ray_grid_xy(npts_per_ray=npts_per_ray)
        self.xxyy_xiaobiao = np.concatenate((xx[:, :, np.newaxis], yy[:, :, np.newaxis]), axis=2)
        aa, rr = gen_ray_grid_ar(npts_per_ray=npts_per_ray)
        self.rr_xiaobiao = rr
        self.aarr_xiaobiao = np.concatenate((aa[:, :, np.newaxis], rr[:, :, np.newaxis]), axis=2)
        
        XX_infer_xiaobiao, _ = dass.gen_X_data(daR, q, self.angles_xiaobiao, npts_per_ray, type='xiaobiao')
        X_infer_xiaobiao = dass.X_to_train_shape(XX_infer_xiaobiao)
        self.X_mu_xiaobiao, self.X_std_xiaobiao = dann.get_norm(X_infer_xiaobiao, eps=1e-5)
        
        if method is 'xiaobiao':
            self._rr = self.rr_xiaobiao
            self._angles = self.angles_xiaobiao
            self.X_mu = self.X_mu_xiaobiao
            self.X_std = self.X_std_xiaobiao
        elif method is 'daniel':
            self._rr = self.rr_daniel
            self._angles = self.angles_daniel
            self.X_mu = self.X_mu_daniel
            self.X_std = self.X_std_daniel
        else:
            raise NotImplementedError
            
        # MA part
        self._spos = gen_spos()
        self._momentum = gen_momentum()
        ss, mm = gen_grid_sm()
        self._mm = mm
        self.ssmm = np.concatenate((ss[:, :, np.newaxis], mm[:, :, np.newaxis]), axis=2)

    def set_fn_da(self, fn_da):
        self._fn_da = fn_da
        
    def set_fn_ma(self, fn_ma):
        self._fn_ma = fn_ma
        
    def fn_obj_da(self, x):
        XX, _ = self.gen_X_data(x, self._exact)
        X = dass.X_to_train_shape(XX)
        Y = self._fn_da(X)

        obj_pred, _ = dass.pred_to_obj(
            Y.T, self.rr_xiaobiao, self.angles_xiaobiao, obj_scaled=None,
            da_thresh=self._da_thresh, method=self._da_method)

        return obj_pred[:, None]
    
    def fn_obj_ma(self, x):
        _, XX = self.gen_X_data(x)
        X = dass.X_to_train_shape(XX)
        Y = self._fn_ma(X)

        obj_pred, _ = dass.pred_to_ma(
            Y.T, self._mm, self._spos, obj_scaled=None,
            ma_thresh=self._ma_thresh, method=self._ma_method)

        return obj_pred[:, None]

    def gen_X_data(self, q, exact=False):
        # q: (n, 4)
        
        nq, _ = q.shape

        if self._method is 'xiaobiao':
            nrays = 19
            npts_per_ray = 72

            qq = np.repeat(q[:, np.newaxis, :], nrays, 1)
            qq2 = np.repeat(qq[:, :, np.newaxis, :], npts_per_ray, 2)
            
            if exact:
                xxyy2 = np.repeat(self.xxyy_xiaobiao[np.newaxis, :, :], nq, 0)

                XX_DA = np.concatenate([qq2, xxyy2], axis=-1)
            else:
                aarr2 = np.repeat(self.aarr_xiaobiao[np.newaxis, :, :], nq, 0)

                XX_DA = np.concatenate([qq2, aarr2], axis=-1)
        elif self._method is 'daniel':
            nrays = 19
            npts_per_ray = 29

            qq = np.repeat(q[:, np.newaxis, :], nrays, 1)
            qq2 = np.repeat(qq[:, :, np.newaxis, :], npts_per_ray, 2)
            
            aarr2 = np.repeat(self.aarr_daniel[np.newaxis, :, :], nq, 0)
            
            XX_DA = np.concatenate([qq2, aarr2], axis=-1)
        else:
            raise NotImplementedError
            
        # MA
        nspos = len(self._spos)
        nmom = len(self._momentum)

        qq = np.repeat(q[:, np.newaxis, :], nspos, 1)
        qq2 = np.repeat(qq[:, :, np.newaxis, :], nmom, 2)

        ssmm2 = np.repeat(self.ssmm[np.newaxis, :, :], nq, 0)

        XX_MA = np.concatenate([qq2, ssmm2], axis=-1)

        return XX_DA, XX_MA
    
    def get_obj_scaled(self, x):
        # x: (n, 6)
        
        return None  # ignore obj_scaled for now (obj_scaled = 1.0)
        
        n, _ = x.shape
        
        s0 = np.array([[0.8578286, -162.44251681]])
        obj_scaled = np.repeat(s0, n, 0)
        
        return obj_scaled


    def _evaluate(self, x, out, *args, **kwargs):
        # x: (n, 4)
        
        da_pred = self.fn_obj_da(x)
        ma_pred = self.fn_obj_ma(x)

        out["F"] = np.hstack((da_pred, ma_pred))


def run_opt_top(problem, pop_size=100, n_gen=200, top_percentage=0.1, verbose=1):
    
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   save_history=True,
                   verbose=verbose)
    
    return get_top_solutions(res, top_percentage)


def run_opt(problem, pop_size=100, n_gen=200, sel_size=100, verbose=1):
    
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   seed=1,
                   save_history=True,
                   verbose=verbose)
    
    return sel_random_solutions(res, sel_size)


def plot_da(a, r, p, title='', show_pos=True, daX=None, daY=None,
            save=None, lim=None, mask=None, use_xy=False):
    fig, ax = plt.subplots()

    if not use_xy:
        x = r * np.cos(a)
        y = r * np.sin(a)
    else:  # if use exact map from matlab
        x = a
        y = r
    z = p.flatten()

    # Create a grid where we want to interpolate data
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z values on the created grid
    # zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = griddata((x, y), z, (xi, yi), method='linear')
    # zi = griddata((x, y), z, (xi, yi), method='nearest')
    zi = np.nan_to_num(zi)

    # Plotting
    img = ax.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()],
                    origin='lower', cmap='Blues', aspect='auto')
    # For a contour map (which can also look like a heatmap), use contourf:
    # plt.contourf(xi, yi, zi, levels=100, cmap='hot')
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    subcolor = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    # plt.plot(daX[idx, :], daY[idx, :], '--o', color=color)
    # plt.plot(daX[idx, :], daY[idx, :], '--o', color='w')
    
    if show_pos:
        if mask is not None:
            ax.plot(x[mask], y[mask], '.', color=color)
        else:
            ax.plot(x, y, '.', color=color)
        
    if daX is not None:
        obj = dass.calc_obj(daX, daY, None)[0]
        ax.plot(daX[0], daY[0], '-o', color=subcolor, label=f'DA = {-obj:.2f}')

    fig.colorbar(img, label='Number of survived turns')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    if lim is not None:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
    
    if daX is not None:
        ax.legend(loc=0)
        
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=200)
    
    plt.show()

    
def plot_da_at_index(idx, y_pred, rr, XX, daX=None, daY=None, title=None,
                     show_pos=True, save=None, lim=None, daR_mask=None, use_xy=False):
    nsamp, nex = y_pred.shape
    nrays, npts_per_ray = rr.shape
    nq = int(nex / (nrays * npts_per_ray))

    turns_pred = dass.Y_to_qar_shape(y_pred[0, :], nq, nrays, npts_per_ray)

    XX_sel = XX[idx]
    turns_sel = turns_pred[idx]

    a_sel = XX_sel[:, :, -2].flatten()
    r_sel = XX_sel[:, :, -1].flatten()
    y_sel = turns_sel.flatten()
    
    if daR_mask is not None:
        mask = daR_mask[idx].flatten()
    else:
        mask = None
    
    if title is None:
        title = f'DA for Config {idx}'
        
    if daX is not None:
        _daX = daX[idx:idx + 1]
        _daY = daY[idx:idx + 1]
    else:
        _daX, _daY = None, None

    plot_da(a_sel, r_sel, y_sel, title=title, show_pos=show_pos,
            daX=_daX, daY=_daY, save=save, lim=lim, mask=mask, use_xy=use_xy)


def plot_ma(x, y, p, title='', show_pos=True, spos=None, maP=None, maM=None,
            save=None, lim=None, mask=None):
    fig, ax = plt.subplots()

    z = p.flatten()

    # Create a grid where we want to interpolate data
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z values on the created grid
    # zi = griddata((x, y), z, (xi, yi), method='cubic')
    zi = griddata((x, y), z, (xi, yi), method='linear')
    # zi = griddata((x, y), z, (xi, yi), method='nearest')
    zi = np.nan_to_num(zi)

    # Plotting
    img = ax.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()],
                    origin='lower', cmap='Reds', aspect='auto')
    # For a contour map (which can also look like a heatmap), use contourf:
    # plt.contourf(xi, yi, zi, levels=100, cmap='hot')
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    subcolor = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
    # plt.plot(daX[idx, :], daY[idx, :], '--o', color=color)
    # plt.plot(daX[idx, :], daY[idx, :], '--o', color='w')
    
    if show_pos:
        if mask is not None:
            ax.plot(x[mask], y[mask], '.', color=color)
        else:
            ax.plot(x, y, '.', color=color)
        
    if maP is not None:
        obj = dass.calc_ma(spos, maP, maM, None)[0]
        ax.plot(spos, maP[0], '-o', color=subcolor, label=f'MA = {-obj:.2f}')
        ax.plot(spos, maM[0], '-o', color=subcolor)

    fig.colorbar(img, label='Number of survived turns')
    ax.set_xlabel('spos')
    ax.set_ylabel('momentum')
    ax.set_title(title)
    
    if lim is not None:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
    
    if maP is not None:
        ax.legend(loc=0)
        
    if save:
        fig.savefig(save, bbox_inches='tight', dpi=200)
    
    plt.show()

    
def plot_ma_at_index(idx, y_pred, mm, XX, spos, maP=None, maM=None, title=None,
                     show_pos=True, save=None, lim=None, ma_mask=None):
    nsamp, nex = y_pred.shape
    nspos, nmom = mm.shape
    nq = int(nex / (nspos * nmom))

    turns_pred = dass.Y_to_qar_shape(y_pred[0, :], nq, nspos, nmom)

    XX_sel = XX[idx]
    turns_sel = turns_pred[idx]

    i_sel = XX_sel[:, :, -2].flatten().astype(int)
    s_sel = spos[i_sel - 1]
    m_sel = XX_sel[:, :, -1].flatten()
    y_sel = turns_sel.flatten()
    
    if ma_mask is not None:
        mask = ma_mask[idx].flatten()
    else:
        mask = None
    
    if title is None:
        title = f'MA for Config {idx}'
        
    if maP is not None:
        _maP = maP[idx:idx + 1]
        _maM = maM[idx:idx + 1]
    else:
        _maP, _maM = None, None

    plot_ma(s_sel, m_sel, y_sel, title=title, show_pos=show_pos,
            spos=spos, maP=_maP, maM=_maM, save=save, lim=lim, mask=mask)
