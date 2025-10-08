from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import scipy.io as scio
from .da_utils import gen_ray_grid_ar
from .dama_resources import get_ssrlx_data


def load_dat(mypath=''):
    """
    Load SSRL data.

    Note: mypath parameter is kept for backward compatibility but is now ignored.
    Uses centralized resource management instead.
    """
    myfile = get_ssrlx_data()
    matdat = scio.loadmat(str(myfile))

    dat = {}
    dat['pf'] = matdat['ds']['pf'][0][0]
    # dat['k2'] = matdat['ds']['k2'][0][0]
    dat['obj'] = matdat['ds']['obj'][0][0]
    dat['obj_scaled'] = matdat['ds']['obj_scaled'][0][0]
    dat['daX'] = matdat['ds']['daX'][0][0]
    dat['daY'] = matdat['ds']['daY'][0][0]

    return (dat)


def gen_X_data(daR, q, angles, npts_per_ray, type='train'):
    nq, nrays = daR.shape
    if type is 'train':
        # r = gen_ray_pts(daR, npts_per_ray)
        r = gen_ray_pts_center(daR, npts_per_ray)
        rr, aa = np.meshgrid(r, angles)
    elif type is 'inference':
        rr = gen_ray_pts_var(daR, npts_per_ray)
        aa = np.repeat(angles[:, np.newaxis], npts_per_ray, 1)
    elif type is 'xiaobiao':
        aa, rr = gen_ray_grid_ar()

    # reshape polar coordinates
    aarr = np.concatenate((aa[:, :, np.newaxis], rr[:, :, np.newaxis]), axis=2)
    aarr2 = np.repeat(aarr[np.newaxis, :, :], nq, 0)

    # reshape control variables
    qq = np.repeat(q[:, np.newaxis, :], nrays, 1)
    qq2 = np.repeat(qq[:, :, np.newaxis, :], npts_per_ray, 2)

    # combine into input vector
    XX = np.concatenate([qq2, aarr2], axis=-1)

    return (XX, rr)


def gen_turns_ssrl_old(rr, daR, delta=0.1):
    # r is a vector of particle radii
    # daR is the corresponding value of sqrt(daX**2 + daY**2) for each value of q
    # q are the control values for each sample

    nq, nrays = daR.shape
    npts_per_ray = rr.shape[1]
    turns = np.zeros([nq, nrays, npts_per_ray])
    for j in range(nq):
        for a in range(nrays):
            # number of turns that particles survive at r
            turns_ja = 2 * sigmoid((daR[j, a] - rr[a, :]) / daR[j, a])
            turns_ja[turns_ja > 1 + delta] = 1 + delta

            turns[j, a, :] = turns_ja

    return (turns)


def gen_turns_ssrl(rr, daR, sigmoid_a = 100, boundary_val = 0.95):
    # r is a vector of particle radii
    # daR is the corresponding value of sqrt(daX**2 + daY**2) for each value of q
    # q are the control values for each sample
    # boundary_val defines the DA boundary the ratio of turns to max simulated turns

    shift = -np.log(1/boundary_val - 1) / sigmoid_a
    print('shift is %.3f' % shift)
    nq, nrays = daR.shape
    npts_per_ray = rr.shape[1]
    turns = np.zeros([nq, nrays, npts_per_ray])
    for j in range(nq):
        for a in range(nrays):
            # number of turns that particles survive at r
            turns_ja = sigmoid((daR[j, a] - rr[a, :]) / daR[j, a] + shift, a = sigmoid_a)

            turns[j, a, :] = turns_ja

    return (turns)


def gen_ray_angles(daX, daY, samp=0):
    # gen ray angles as defined by ground truth daX and daY

    nrays = daX.shape[1]
    angles = np.zeros(nrays)
    for jj in range(nrays):
        angles[jj] = np.arctan2(daY[samp, jj], daX[samp, jj])

    return (angles)


def gen_ray_pts(daR, npts_per_ray, range_scale=2):
    # gen ray pts assuming the same range for all rays
    # could also gen ray pts with variable range (commented line)

    # nq,nrays = daR.shape
    # r_max = range_scale * np.max(daR, axis=0)
    r_max = range_scale * np.max(daR)
    r = np.linspace(0, r_max, npts_per_ray)

    return (r)


def gen_ray_pts_center(daR, npts_per_ray, range_scale=0.1):
    # gen ray pts assuming the same range for all rays centered on aperture

    # nq,nrays = daR.shape
    # r_max = range_scale * np.max(daR, axis=0)
    r_min = (1 - range_scale) * np.min(daR)
    r_max = (1 + range_scale) * np.max(daR)
    r = np.linspace(0, r_max, npts_per_ray)

    return (r)


def gen_ray_pts_var(daR, npts_per_ray=29, range_scale=0.1):
    # gen ray pts for each ray individually

    nq, nrays = daR.shape
    rr = np.zeros([nrays, npts_per_ray])
    for j in range(nrays):
        daRj = daR[:, j]
        rdiff_vec = daRj - daRj[0]
        rdiff = np.min(np.abs(rdiff_vec[rdiff_vec > 0]))

        rmin = np.min(daRj)
        rmax = np.max(daRj)
        rsteps = int((rmax - rmin) / (2 * rdiff))
        #        print(rsteps)

        r0j = rmin + rsteps * rdiff

        half_range = int((npts_per_ray - 1) / 2) * rdiff
        rj = np.linspace(r0j - half_range, r0j + half_range, npts_per_ray)

        rr[j, :] = rj

    return (rr)


def calc_daXY(turns, rr, angles, da_thresh=0.5, method=0):
    # given a pred for the number of turns each particle survives (turns), calculate corresponding daX, daY
    # method: 0 -> Daniel's method; 1 -> Xiaobiao's method
    
    _turns = turns.copy()  # copy and do not modify the original array

    nq, nrays, npts_per_ray = _turns.shape
    daR_pred = np.zeros([nq, nrays])
    for k in range(nq):
        for jj in range(nrays):
            _turns[k, jj, 0] = 1  # force r=0 to be "inside" the aperture
            if method:
                try:
                    idx = np.where(_turns[k, jj, :] < da_thresh)[0][0] - 1
                except:
                    idx = -1
            else:
                idx = np.where(_turns[k, jj, :] >= da_thresh)[0][-1]
            daR_pred[k, jj] = rr[jj, idx]

    daX_pred = np.cos(angles[np.newaxis, :]) * daR_pred
    daY_pred = np.sin(angles[np.newaxis, :]) * daR_pred

    return (daX_pred, daY_pred)


def calc_daXY_ex(turns, rr, angles, da_thresh=0.5, method=0):
    # given a pred for the number of turns each particle survives (turns), calculate corresponding daX, daY
    # method: 0 -> Daniel's method; 1 -> Xiaobiao's method
    
    _turns = turns.copy()  # copy and do not modify the original array

    nq, nrays, npts_per_ray = _turns.shape
    daR_pred = np.zeros([nq, nrays])
    daR_idx_pred = np.zeros([nq, nrays])
    for k in range(nq):
        for jj in range(nrays):
            _turns[k, jj, 0] = 1  # force r=0 to be "inside" the aperture
            if method:
                try:
                    idx = np.where(_turns[k, jj, :] < da_thresh)[0][0] - 1
                except:
                    idx = -1
            else:
                idx = np.where(_turns[k, jj, :] >= da_thresh)[0][-1]
            daR_pred[k, jj] = rr[jj, idx]
            daR_idx_pred[k, jj] = idx

    daX_pred = np.cos(angles[np.newaxis, :]) * daR_pred
    daY_pred = np.sin(angles[np.newaxis, :]) * daR_pred

    return (daX_pred, daY_pred, daR_idx_pred)


def calc_obj(daX, daY, obj_scaled):
    # calculate the objective given prediction of daX and daY and given the (secretly known) obj_scaled
    # eventually should model the parameters used to calculate obj_scaled as well

    nq, nrays = daX.shape
    xyarea = np.zeros(nq)

    for k in range(nq):
        for jj in range(1, nrays):
            scale = 1
            if daX[k, jj] < 0:  # add more weight to the negative side
                scale = 2
            else:
                scale = 1
            xyarea[k] = xyarea[k] + scale * 0.5 * np.sqrt(daX[k, jj] ** 2 + daY[k, jj] ** 2) * np.sqrt(
                daX[k, jj - 1] ** 2 + daY[k, jj - 1] ** 2) * np.sin(
                np.arctan2(daY[k, jj], daX[k, jj]) - np.arctan2(daY[k, jj - 1], daX[k, jj - 1]));

    if obj_scaled is not None:
        obj = -1e6 * xyarea / obj_scaled[:, 0]
    else:
        obj = -1e6 * xyarea
    #    plt.plot(obj[:,0])
    #    plt.plot(obj,'--')
    #    plt.xlabel('sample')
    #    plt.show()

    return (obj)


def calc_maPM(turns, mm, spos, ma_thresh=0.5, method=0):
    # given a pred for the number of turns each particle survives (turns), calculate corresponding daX, daY
    # method: 0 -> Daniel's method; 1 -> Xiaobiao's method; 2 -> the wrongly Xiaobiao's method
    
    _turns = turns.copy()  # copy and do not modify the original array

    nq, nspos, nmom = _turns.shape
    maP_pred = np.zeros([nq, nspos])
    maM_pred = np.zeros([nq, nspos])
    mid = nmom // 2
    for k in range(nq):
        for jj in range(nspos):
            _turns[k, jj, mid] = 1  # force center point to be "inside" the aperture
            if method == 1:
                try:
                    idx_P = np.where(_turns[k, jj, mid:] < ma_thresh)[0][0] - 1
                except:
                    idx_P = -1
                try:
                    idx_M = np.where(_turns[k, jj, :mid] < ma_thresh)[0][-1] + 1
                except:
                    idx_M = 0
            elif method == 2:
                try:
                    idx_P = np.where(_turns[k, jj, mid:] < ma_thresh)[0][0]
                except:
                    idx_P = -1
                try:
                    idx_M = np.where(_turns[k, jj, :mid] < ma_thresh)[0][-1]
                except:
                    idx_M = 0
            else:
                idx_P = np.where(_turns[k, jj, mid:] >= ma_thresh)[0][-1]
                idx_M = np.where(_turns[k, jj, :mid] >= ma_thresh)[0][0]
            maP_pred[k, jj] = mm[jj, idx_P + mid]
            maM_pred[k, jj] = mm[jj, idx_M]

    return (maP_pred, maM_pred)


def calc_maPM_ex(turns, mm, spos, ma_thresh=0.5, method=0):
    # given a pred for the number of turns each particle survives (turns), calculate corresponding daX, daY
    # method: 0 -> Daniel's method; 1 -> Xiaobiao's method; 2 -> the wrongly Xiaobiao's method
    
    _turns = turns.copy()  # copy and do not modify the original array

    nq, nspos, nmom = _turns.shape
    maP_pred = np.zeros([nq, nspos])
    maM_pred = np.zeros([nq, nspos])
    maPM_idx_pred = np.zeros([nq, nspos, 2])
    mid = nmom // 2
    for k in range(nq):
        for jj in range(nspos):
            _turns[k, jj, mid] = 1  # force center point to be "inside" the aperture
            if method == 1:
                try:
                    idx_P = np.where(_turns[k, jj, mid:] < ma_thresh)[0][0] - 1
                except:
                    idx_P = -1
                try:
                    idx_M = np.where(_turns[k, jj, :mid] < ma_thresh)[0][-1] + 1
                except:
                    idx_M = 0
            elif method == 2:
                try:
                    idx_P = np.where(_turns[k, jj, mid:] < ma_thresh)[0][0]
                except:
                    idx_P = -1
                try:
                    idx_M = np.where(_turns[k, jj, :mid] < ma_thresh)[0][-1]
                except:
                    idx_M = 0
            else:
                idx_P = np.where(_turns[k, jj, mid:] >= ma_thresh)[0][-1]
                idx_M = np.where(_turns[k, jj, :mid] >= ma_thresh)[0][0]
            maP_pred[k, jj] = mm[jj, idx_P + mid]
            maM_pred[k, jj] = mm[jj, idx_M]
            maPM_idx_pred[k, jj, 0] = idx_P + mid
            maPM_idx_pred[k, jj, 1] = idx_M

    return (maP_pred, maM_pred, maPM_idx_pred)


def calc_ma(spos, maP, maM, obj_scaled):
    # calculate the objective given prediction of daX and daY and given the (secretly known) obj_scaled
    # eventually should model the parameters used to calculate obj_scaled as well

    nq, nspos = maP.shape
    xyarea = np.zeros(nq)

    for k in range(nq):
        xyarea[k] = np.sum(((maP[k, 1:] + maP[k, :-1]) / 2 - (maM[k, 1:] + maM[k, :-1]) / 2) * (spos[1:] - spos[:-1]))

    if obj_scaled is not None:
        obj = -1 * xyarea / obj_scaled[:, 0]
    else:
        obj = -1 * xyarea
    #    plt.plot(obj[:,0])
    #    plt.plot(obj,'--')
    #    plt.xlabel('sample')
    #    plt.show()

    return (obj)


def pred_to_obj(y_pred, rr, angles, obj_scaled, da_thresh=0.5, method=0):
    nsamp, nex = y_pred.shape
    nrays, npts_per_ray = rr.shape
    nq = int(nex / (nrays * npts_per_ray))
    obj_pred_reg = np.zeros([nsamp, nq])
    for j in range(nsamp):
        turns_pred = Y_to_qar_shape(y_pred[j, :], nq, nrays, npts_per_ray)
        daX_pred, daY_pred = calc_daXY(turns_pred, rr, angles,
                                       da_thresh=da_thresh, method=method)
        #    plt.plot(daX[:,10],daX_pred[:,10],'.'); plt.show()
        #    for k in range(19):
        #        plt.plot(daY[:,k],daY_pred[:,k],'.');
        #    plt.show()
        #    daR_pred = np.sqrt(daX_pred**2 + daY_pred**2)
        #    plt.plot(daR,daR_pred,'.'); plt.show()

        obj_pred_reg[j, :] = calc_obj(daX_pred, daY_pred, obj_scaled)

    obj_pred = np.mean(obj_pred_reg, axis=0)
    obj_pred_std = np.std(obj_pred_reg, axis=0)

    return (obj_pred, obj_pred_std)


def pred_to_ma(y_pred, mm, spos, obj_scaled, ma_thresh=0.5, method=0):
    nsamp, nex = y_pred.shape
    nspos, nmom = mm.shape
    nq = int(nex / (nspos * nmom))
    obj_pred_reg = np.zeros([nsamp, nq])
    for j in range(nsamp):
        turns_pred = Y_to_qar_shape(y_pred[j, :], nq, nspos, nmom)
        maP_pred, maM_pred = calc_maPM(turns_pred, mm, spos,
                                       ma_thresh=ma_thresh, method=method)
        #    plt.plot(daX[:,10],daX_pred[:,10],'.'); plt.show()
        #    for k in range(19):
        #        plt.plot(daY[:,k],daY_pred[:,k],'.');
        #    plt.show()
        #    daR_pred = np.sqrt(daX_pred**2 + daY_pred**2)
        #    plt.plot(daR,daR_pred,'.'); plt.show()

        obj_pred_reg[j, :] = calc_ma(spos, maP_pred, maM_pred, obj_scaled)

    obj_pred = np.mean(obj_pred_reg, axis=0)
    obj_pred_std = np.std(obj_pred_reg, axis=0)

    return (obj_pred, obj_pred_std)


def pred_to_obj_ex(y_pred, rr, angles, obj_scaled, da_thresh=0.5, method=0):
    nsamp, nex = y_pred.shape
    nrays, npts_per_ray = rr.shape
    nq = int(nex / (nrays * npts_per_ray))
    obj_pred_reg = np.zeros([nsamp, nq])
    for j in range(nsamp):
        turns_pred = Y_to_qar_shape(y_pred[j, :], nq, nrays, npts_per_ray)
        daX_pred, daY_pred = calc_daXY(turns_pred, rr, angles,
                                       da_thresh=da_thresh, method=method)
        #    plt.plot(daX[:,10],daX_pred[:,10],'.'); plt.show()
        #    for k in range(19):
        #        plt.plot(daY[:,k],daY_pred[:,k],'.');
        #    plt.show()
        #    daR_pred = np.sqrt(daX_pred**2 + daY_pred**2)
        #    plt.plot(daR,daR_pred,'.'); plt.show()

        obj_pred_reg[j, :] = calc_obj(daX_pred, daY_pred, obj_scaled)

    obj_pred = np.mean(obj_pred_reg, axis=0)
    obj_pred_std = np.std(obj_pred_reg, axis=0)

    return (obj_pred, obj_pred_std, daX_pred, daY_pred)


def calc_da_vec(turns):
    # appoximate DA by fraction of particles that survive (assuming random/uniform sampling)

    survive = turns >= 1
    num_survive = np.sum(np.sum(survive, axis=1), axis=1)
    n_part = turns.shape[0] * turns.shape[1]
    da = num_survive / n_part

    da_full = da[:, np.newaxis, np.newaxis] * np.ones(turns.shape)

    return (da_full)


def sigmoid(x, a=10):
    return (1 / (1 + np.exp(-a * x)))


def make_rays(nrays, npts_per_ray):
    a = np.linspace(0, np.pi, nrays)
    r = np.linspace(0, 1, npts_per_ray)
    R, A = np.meshgrid(r, angles)
    X = np.cos(A) * R
    Y = np.sin(A) * R

    return (X, Y)


def X_to_train_shape(X):
    nq, nrays, npts_per_ray, nfeat = X.shape
    X_train = np.reshape(X, [nq * nrays * npts_per_ray, nfeat], order='F')

    return (X_train)


def X_to_qar_shape(X, nq, nrays, npts_per_ray, nfeat):
    X_qar = np.reshape(X, [nq, nrays, npts_per_ray, nfeat], order='F')

    return (X_qar)


def Y_to_train_shape(Y):
    nq, nrays, npts_per_ray = Y.shape
    Y_train = np.reshape(Y, nq * nrays * npts_per_ray, order='F')

    return (Y_train)


def Y_to_qar_shape(Y, nq, nrays, npts_per_ray):
    Y_qar = np.reshape(Y, [nq, nrays, npts_per_ray], order='F')

    return (Y_qar)


def x_to_train_shape(x):  # x is X without the first dim
    nrays, npts_per_ray, nfeat = x.shape
    x_train = np.reshape(x, [nrays * npts_per_ray, nfeat], order='F')

    return (x_train)


def batch_acquire(obj_pred, obj_pred_std, y_pred, nq, nrays, npts_per_ray_pred, acquire_batch_size=100,
                  obj_percentile=95, obj_std_percentile=75, turns_std_percentile=75):
    # calculate predicted turn statistics
    y_pred_mu = np.mean(y_pred, axis=0)
    y_pred_std = np.std(y_pred, axis=0)
    turns_pred_mu = Y_to_qar_shape(y_pred_mu, nq, nrays, npts_per_ray_pred)
    turns_pred_std = Y_to_qar_shape(y_pred_std, nq, nrays, npts_per_ray_pred)

    # select top solutions
    obj_thresh = np.percentile(-obj_pred, obj_percentile)
    top_q = np.where(-obj_pred > obj_thresh)[0]

    obj_std_thresh = np.percentile(obj_pred_std[top_q], obj_std_percentile)
    top_q_unc = top_q[np.where(obj_pred_std[top_q] > obj_std_thresh)]

    # select most uncertain from top solutions
    turns_std_thresh = np.percentile(turns_pred_std[top_q_unc, :, :], turns_std_percentile)
    top_pts = np.where(turns_pred_std[top_q_unc, :, :] > turns_std_thresh)

    n_top_pts = top_pts[0].shape[0]
    acquire_batch_size = np.min([acquire_batch_size, n_top_pts])
    batch_ind = np.random.choice(n_top_pts, acquire_batch_size, replace=False)
    batch = np.zeros([3, acquire_batch_size]).astype('int')
    for j in range(3):
        batch[j, :] = top_pts[j][batch_ind]

    for k in range(acquire_batch_size):
        batch[0, k] = top_q_unc[batch[0, k]]

    # batch = turns_subset[top_pts[batch_ind[0],batch_ind[1],batch_ind[2]]

    return batch


def batch_acquire_thompson(obj_pred, obj_pred_std, y_pred, nq, nrays, npts_per_ray_infer, acquire_batch_size=100,
                  obj_percentile=95, turns_min=.95, turns_max=1.05):

    # select top solutions
    obj_thresh = np.percentile(-obj_pred, obj_percentile)
    top_q = np.where(-obj_pred > obj_thresh)[0]

    # select points near the boundary
    turns_pred = Y_to_qar_shape(y_pred, nq, nrays, npts_per_ray_infer)
    top_pts = np.where(np.logical_and(turns_pred[top_q, :, :] > turns_min, turns_pred[top_q, :, :] < turns_max))

    n_top_pts = top_pts[0].shape[0]
    acquire_batch_size = np.min([acquire_batch_size, n_top_pts])
    batch_ind = np.random.choice(n_top_pts, acquire_batch_size, replace=False)
    batch = np.zeros([3, acquire_batch_size]).astype('int')
    for j in range(3):
        batch[j, :] = top_pts[j][batch_ind]

    for k in range(acquire_batch_size):
        batch[0, k] = top_q[batch[0, k]]

    # batch = turns_subset[top_pts[batch_ind[0],batch_ind[1],batch_ind[2]]

    return batch


def run_top_k_overlap(obj_pred, obj_gt, pred_percentile_scan, gt_percentile=99):
    n_percentile = pred_percentile_scan.shape[0]
    overlap_scan = np.zeros(n_percentile)
    for j in range(n_percentile):
        overlap_scan[j] = top_k_overlap_metric(obj_pred, obj_gt, pred_percentile=pred_percentile_scan[j],
                                               gt_percentile=gt_percentile)

    return overlap_scan


def top_k_overlap_metric(obj_pred, obj_gt, pred_percentile=95, gt_percentile=99):
    obj_pred_thresh = np.percentile(-obj_pred, pred_percentile)
    obj_gt_thresh = np.percentile(-obj_gt, gt_percentile)
    top_q = np.where(np.logical_and(-obj_pred > obj_pred_thresh, -obj_gt > obj_gt_thresh))[0]
    n = (100 - gt_percentile) / 100 * obj_pred.shape[0]
    overlap = top_q.shape[0] / n

    return overlap


def viz_discontinuity(q, da):
    # a0 = 10
    npts = q.shape[0]
    n = 6000
    nrays = da.shape[1]
    dist = np.zeros(n)
    daj = np.zeros(n)
    for j in range(n):
        k = np.random.choice(npts, 2, replace=False)
        l = np.random.choice(nrays, 1)
        #        dist[j] = np.sum(np.abs(q[k[0],:] - q[k[1],:]))
        #        daj[j] = np.abs(daX[k[0],l] - daX[k[1],l])
        dist[j] = np.sum(np.abs(q[j + 1, :] - q[j, :]))
        daj[j] = np.abs(daX[j + 1, l] - daX[j, l])

    plt.plot(dist, daj, '.');
    plt.show()


def umap_obj_fit(q):
    import umap
    reducer = umap.UMAP(random_state=42)
    reducer.fit(q)
    embed = reducer.transform(q)
    return embed


def umap_obj_plot(embed, obj_pred, obj_gt, pred_percentile=95):
    obj_pred_thresh = np.percentile(-obj_pred, pred_percentile)
    obj_gt_thresh = np.percentile(-obj_gt, pred_percentile)

    plt.plot(embed[:, 0], embed[:, 1], 'o', color='grey', markersize=2)
    top_q_pred = np.where(-obj_pred > obj_pred_thresh)[0]
    top_q_gt = np.where(-obj_gt > obj_gt_thresh)[0]
    plt.plot(embed[top_q_gt, 0], embed[top_q_gt, 1], 'o', color='red', markersize=3)
    plt.plot(embed[top_q_pred, 0], embed[top_q_pred, 1], 'o', color='blue', markersize=3)
    plt.title('Top %d percentile configurations' % pred_percentile)
    plt.legend(['all configs','top %d percent' % pred_percentile,'predicted top %d percent' % pred_percentile])
    plt.show()
    
    
def get_srange():
    # sext ranges
    return np.array([
        [135, 195],
        [135, 195],
        [-195, -135],
        [-195, -135],
        [-195, -135],
        [-195, -135]
    ]).T


# Convert the data to the simulatable format
def format_data_sim(data, q_opt=None, dim_selection=[0, 1, 2, 3, 4, 5]):
    # data: [si, ..., sj, theta, r], note that column number is equal to len(dim_selection) + 2
    # q_opt: optimal solution to base on
    # dim_selection: [i, ..., j], which dimensions to vary from q_opt
    # formatted_data for sim: [s0_norm, ..., s5_norm, daX, daY]
    data_formatted = np.zeros((data.shape[0], 8))
    if q_opt is not None:
        data_formatted[:, :6] = q_opt
    
    # Fill in the selected sext
    for i, dim in enumerate(dim_selection):
        data_formatted[:, dim] = data[:, i]
    srange = get_srange()
    data_formatted[:, :6] = (data_formatted[:, :6] - srange[0, :]) / (srange[1, :] - srange[0, :])
    
    # Fill in daX and daY
    theta = data[:, -2]
    r = data[:, -1]
    data_formatted[:, 6] = r * np.cos(theta)
    data_formatted[:, 7] = r * np.sin(theta)
    
    return data_formatted


def load_XY_batch_sim(data_root='data', prefix='loop'):
    with open(os.path.join(data_root, f'{prefix}_X.npy'), 'rb') as f:
        X_batch = np.load(f)
        X_batch_formatted = np.load(f)
        
    num_files = len([n for n in os.listdir(data_root) if n.startswith(f'{prefix}_Y_')])
    Y_batch = []
    for i in range(num_files):
        with open(os.path.join(data_root, f'{prefix}_Y_{i}.npy'), 'rb') as f:
            Y_batch.append(np.load(f))
    Y_batch = np.vstack(Y_batch).flatten() / 1024
    
    return X_batch, X_batch_formatted, Y_batch


def get_Y_batch_sim(X_batch, evaluate, buffer_size=168, data_root='data', prefix='loop'):
    # Only for data w/ 6 sexts
    
    # Check if already simulated
    try:
        _X_batch, _, _Y_batch = load_XY_batch_sim(data_root=data_root, prefix=prefix)
        
        assert _Y_batch.shape[0] == _X_batch.shape[0]
        assert np.allclose(X_batch, _X_batch)
        
        print('Simulated data loaded')
        return _Y_batch
    except:
        pass  # no simulated data available/usable, go ahead and simulate it
    
    # Simulate it
    X_batch_formatted = format_data_sim(X_batch, dim_selection=[0, 1, 2, 3, 4, 5])
    
    os.makedirs(data_root, exist_ok=True)
    with open(os.path.join(data_root, f'{prefix}_X.npy'), 'wb') as f:
        np.save(f, X_batch)
        np.save(f, X_batch_formatted)
    
    count = 0
    while True:
        sc = count * buffer_size
        ec = (count + 1) * buffer_size
        _X = X_batch_formatted[sc:ec]

        if not _X.size:
            print('Simulation completed')
            break

        print(f'Simulating {prefix} data {sc} to {ec - 1}...')

        t0 = time.time()
        _Y = evaluate(_X)
        t1 = time.time()
        with open(os.path.join(data_root, f'{prefix}_Y_{count}.npy'), 'wb') as f:
            np.save(f, _Y)

        print(f'{prefix} data {sc} to {ec - 1} saved. Time elapsed: {t1 - t0:.2f}s')

        count += 1
        
    num_files = int(np.ceil(X_batch_formatted.shape[0] / buffer_size))
    Y_batch = []
    for i in range(num_files):
        with open(os.path.join(data_root, f'{prefix}_Y_{i}.npy'), 'rb') as f:
            Y_batch.append(np.load(f))
    Y_batch = np.vstack(Y_batch).flatten() / 1024
    
    return Y_batch


def format_data_sim_4d(data):
    # data: [s1, s2, s3, s4, theta, r], note that column number is equal to len(dim_selection) + 2
    # formatted_data for sim: [s1, s2, s3, s4, daX, daY]
    data_formatted = np.zeros((data.shape[0], 6))
    data_formatted[:, :4] = data[:, :4]
    
    # Fill in daX and daY
    theta = data[:, -2]
    r = data[:, -1]
    data_formatted[:, 4] = r * np.cos(theta)
    data_formatted[:, 5] = r * np.sin(theta)
    
    return data_formatted


def get_Y_batch_sim_4d(X_batch, evaluate, buffer_size=168, data_root='data', prefix='loop',
                       format_data=True, X_origin=None, verbose=True):
    # Only for data w/ 4 combined sexts
    # X_origin: if provided, save this to file instead
    
    # Check if already simulated
    try:
        _X_batch, _, _Y_batch = load_XY_batch_sim(data_root=data_root, prefix=prefix)
        
        assert _Y_batch.shape[0] == _X_batch.shape[0]
        if X_origin is None:
            assert np.allclose(X_batch, _X_batch)
        else:
            assert np.allclose(X_origin, _X_batch)
        
        if verbose:
            print(f'Simulated data loaded from {prefix} in {data_root}')
        return _Y_batch
    except:
        pass  # no simulated data available/usable, go ahead and simulate it
    
    # Simulate it
    if format_data:
        X_batch_formatted = format_data_sim_4d(X_batch)
    else:
        X_batch_formatted = X_batch
    
    os.makedirs(data_root, exist_ok=True)
    with open(os.path.join(data_root, f'{prefix}_X.npy'), 'wb') as f:
        if X_origin is None:
            np.save(f, X_batch)
        else:
            np.save(f, X_origin)
        np.save(f, X_batch_formatted)
    
    count = 0
    while True:
        sc = count * buffer_size
        ec = (count + 1) * buffer_size
        _X = X_batch_formatted[sc:ec]

        if not _X.size:
            if verbose:
                print('Simulation completed')
            break

        if verbose:
            print(f'Simulating {prefix} data {sc} to {ec - 1}...')

        t0 = time.time()
        _Y = evaluate(_X)
        t1 = time.time()
        with open(os.path.join(data_root, f'{prefix}_Y_{count}.npy'), 'wb') as f:
            np.save(f, _Y)

        if verbose:
            print(f'{prefix} data {sc} to {ec - 1} saved. Time elapsed: {t1 - t0:.2f}s')

        count += 1
        
    num_files = int(np.ceil(X_batch_formatted.shape[0] / buffer_size))
    Y_batch = []
    for i in range(num_files):
        with open(os.path.join(data_root, f'{prefix}_Y_{i}.npy'), 'rb') as f:
            Y_batch.append(np.load(f))
    Y_batch = np.vstack(Y_batch).flatten() / 1024
    
    return Y_batch
