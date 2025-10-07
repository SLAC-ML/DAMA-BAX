import math
import pickle
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import at
from at import (
    Drift, Marker, Quadrupole, Sextupole, Octupole, Multipole,
    Dipole, RFCavity, Lattice,
)


def gen_damap(q,
              a: float = 5,
              npts_per_ray: int = 72,
              x0_min: float = 0.004,
              x0_max: float = 0.025,
              y0_max: float = 0.012):
    """
    Build the DA‐map design matrix for points q.

    Parameters
    ----------
    q : array‐like, shape (nq, nfeat)
        Your “query” features.
    a : float, optional
        Controls the curvature of the ray angles (default=5).
    npts_per_ray : int, optional
        How many points along each ray (default=72).
    x0_min, x0_max, y0_max : float, optional
        Radial scaling parameters (default 0.004, 0.025, 0.012).

    Returns
    -------
    X_train : ndarray, shape (nq * nrays * npts_per_ray, nfeat + 2)
        Stacked [q ‖ (x_offset, y_offset)] for every ray‐point,
        flattened in Fortran order so that it matches your old
        `X_to_train_shape(..., order='F')`.
    """
    q = np.asarray(q)
    nq, nfeat = q.shape

    # 1) generate the 19 ray‐angles (originally from gen_ray_angles)
    x1 = np.arange(10)
    theta1 = (10 - a) * x1 + (a / 9) * x1**2
    x2 = np.arange(1, 10)
    theta2 = 90 + (10 + a) * x2 - (a / 9) * x2**2
    theta = np.concatenate([theta1, theta2]) * np.pi / 180  # shape (19,)

    # 2) radial distances along each ray
    radl = x0_min + np.arange(npts_per_ray) / npts_per_ray * (x0_max - x0_min)
    # make a (19, 72) grid of (r,θ)
    rr, aa = np.meshgrid(radl, theta)  # rr, aa both shape (19, 72)

    # 3) physical offsets in x,y
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa) * (y0_max / x0_max)
    offsets = np.stack([xx, yy], axis=-1)      # shape (19, 72, 2)

    # 4) tack on your q‐features via broadcasting
    q_b = q[:, None, None, :]  # (nq, 1, 1, nfeat)
    offsets_b = np.broadcast_to(offsets[None, :, :, :], (nq, *offsets.shape))  # (nq, 19, 72, 2)

    # 5) concatenate → (nq, 19, 72, nfeat+2)
    X_full = np.concatenate([
        np.broadcast_to(q_b, (nq, *offsets.shape[:2], nfeat)),
        offsets_b
    ], axis=-1)

    # 6) flatten the first three dims into one, preserving Fortran order
    X_train = X_full.reshape(-1, nfeat + 2, order='F')

    return X_train


def make_gen_mamap():
    setup = np.load('setup.npz')

    spos = setup['spos']
    momentum = setup['momentum']

    def gen_mamap(q, spos=spos, momentum=momentum):
        """
        Build the MA‐map design matrix for queries q.

        Parameters
        ----------
        q : array-like, shape (nq, nfeat)
            Your “query” features (e.g. shape (n,4)).
        spos : array-like, shape (nspos,)
            Just used for its length; original code did `s = arange(1, len(spos)+1)`.
        momentum : array-like, shape (nmom,)
            The momentum grid values.

        Returns
        -------
        X_train : ndarray, shape (nq * nspos * nmom, nfeat + 2)
            Stacked [q ‖ (s_index, momentum)] for every grid point,
            flattened in Fortran order to match your old `X_to_train_shape(..., order='F')`.
        """
        # turn inputs into arrays
        q        = np.asarray(q)
        spos     = np.asarray(spos)
        momentum = np.asarray(momentum)

        nq, nfeat   = q.shape
        nspos       = spos.size
        nmom        = momentum.size

        # build the “s” index array (1,2,…,nspos) and mesh it with momentum
        s = np.arange(1, nspos + 1)
        mm, ss = np.meshgrid(momentum, s)   # mm, ss each shape (nspos, nmom)

        # stack them into an (nspos, nmom, 2) offset array
        offsets = np.stack((ss, mm), axis=-1)

        # broadcast q and offsets into (nq, nspos, nmom, *)
        q_exp    = np.broadcast_to(q[:, None, None, :], (nq, nspos, nmom, nfeat))
        off_exp  = np.broadcast_to(offsets[None, :, :, :], (nq, nspos, nmom, 2))

        # concatenate feature-wise, then flatten the first three dims in Fortran order
        X_full   = np.concatenate((q_exp, off_exp), axis=-1)
        X_train  = X_full.reshape(-1, nfeat + 2, order='F')

        return X_train

    return gen_mamap


def plot_damap(Y):
    """
    Plot one heatmap per sample in Y.

    Parameters
    ----------
    Y : array-like, shape (n*19*72, 1) or (n*19*72,)
        The model outputs you get after evaluating against X.
        Must be divisible by 19*72.

    Produces
    -------
    A matplotlib figure with n subplots, each shaped 72×19 after rotation,
    with true visual aspect ratio preserved and minimal whitespace.
    """
    Y = np.asarray(Y)
    Y_flat = Y.ravel()

    # Constants from gen_damap
    nrays = 19
    npts = 72

    N = Y_flat.size
    if N % (nrays * npts) != 0:
        raise ValueError(f"Y.size ({N}) is not divisible by {nrays}*{npts}")
    n = N // (nrays * npts)

    # Reshape to (n, 19, 72) in Fortran order
    Y_maps = Y_flat.reshape((n, nrays, npts), order='F')

    # Grid layout
    ncols = min(n, 10)
    nrows = math.ceil(n / ncols)

    # After rotation: each map has shape (72, 19)
    # So aspect ratio is height/width = 72 / 19
    aspect_ratio = 72 / 19
    subplot_height = 3.0  # arbitrary base height
    subplot_width = subplot_height / aspect_ratio
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height),
                             squeeze=False,
                             constrained_layout=True)
    fig.patch.set_facecolor('white')

    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        # Rotate (19, 72) → (72, 19) so the ray is vertical
        img = np.rot90(Y_maps[idx], k=1)
        ax.imshow(img, cmap='Blues', aspect='auto')
        ax.set_title(f"Map #{idx+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis('off')

    plt.show()


def plot_mamap(Y):
    """
    Plot one heatmap per sample in Y.

    Parameters
    ----------
    Y : array-like, shape (n*19*72, 1) or (n*19*72,)
        The model outputs you get after evaluating against X.
        Must be divisible by 19*72.

    Produces
    -------
    A matplotlib figure with n subplots, each of size 72×19, using the Blues colormap.
    """
    Y = np.asarray(Y)
    # flatten off any singleton second dim
    Y_flat = Y.ravel()

    # constants from gen_damap
    nrays = 21
    npts  = 49

    N = Y_flat.size
    if N % (nrays * npts) != 0:
        raise ValueError(f"Y.size ({N}) is not divisible by {nrays}*{npts}")
    n = N // (nrays * npts)

    # reshape back into (n, 19, 72) in Fortran order
    Y_maps = Y_flat.reshape((n, nrays, npts), order='F')

    # Grid layout
    ncols = min(n, 10)
    nrows = math.ceil(n / ncols)

    aspect_ratio = npts / nrays
    subplot_height = 3.0  # arbitrary base height
    subplot_width = subplot_height / aspect_ratio
    fig_width = ncols * subplot_width
    fig_height = nrows * subplot_height

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height),
                             squeeze=False,
                             constrained_layout=True)
    fig.patch.set_facecolor('white')

    for idx in range(n):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        # rotate each map (19,72) → (72,19) by 90° CCW
        img = np.rot90(Y_maps[idx], k=1)
        im = ax.imshow(img, cmap='Reds', aspect='equal')
        ax.set_title(f"Map #{idx+1}")
        ax.set_xticks([])
        ax.set_yticks([])

    # turn off any unused subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis('off')

    plt.show()


def find_row_indices(selected, base):
    # For each row in selected, find its index in base

    indices = np.where((base==selected[:, None]).all(-1))[1]

    return indices


def find_element_indices(selected, base):

    differences = np.abs(selected[:, None] - base)
    indices = np.argmin(differences, axis=1)

    return indices


def get_param_group(ring, group):
    # group has shape (1, n)
    indices = [ele[0][0, 0] for ele in group[0, :]]
    K2 = []
    for idx in indices:
        # idx is 1-based in matlab
        element = ring[idx - 1]
        _K2 = element.PolynomB[2]
        K2.append(_K2)

    return np.array(K2)


def set_param_group(ring, group, value):
    # in-place modification!
    # group has shape (1, n)
    indices = [ele[0][0, 0] for ele in group[0, :]]
    for idx in indices:
        # idx is 1-based in matlab
        element = ring[idx - 1]
        element.PolynomB[2] = value


def find_cells(ring, field, value=None):
    if value is None:
        indices = [i for i, elem in enumerate(ring) if hasattr(elem, field)]
    else:
        indices = [i for i, elem in enumerate(ring) if getattr(elem, field, None) == value]

    return indices


def set_cells(ring, field, value, indices=None):
    if indices is None:
        indices = find_cells(ring, field)

    for idx in indices:
        elem = ring[idx]
        setattr(elem, field, value)


def track_DA(ring, dpp, Nturn, pos):
    """
    Simulates tracking for dynamic aperture (DA) evaluation.

    Parameters:
        ring : list
            The accelerator lattice (e.g., loaded from a file) as a list of elements.
        dpp : float
            The momentum deviation.
        Nturn : int
            The number of turns for tracking.
        pos : tuple or list of two floats
            The initial horizontal and vertical positions, e.g. (x, y).

    Returns:
        NturnSurvived : int
            The number of turns that the particle survived.
    """
    # Turn on the radiation cavity with a voltage of 6.0e6.
    icav = find_cells(ring, 'Frequency')
    set_cells(ring, 'Voltage', 6.0e6, icav)
    ring.radiation_on()

    # Find indices for cells containing the 'Frequency' field.
    icav = find_cells(ring, 'Frequency')

    # Find indices for cells whose PassMethod is 'AperturePass'.
    ia = find_cells(ring, 'PassMethod', 'AperturePass')

    # Replace their PassMethod with 'DriftPass'.
    set_cells(ring, 'PassMethod', 'DriftPass', ia)

    cspeed = 2.99792458e8  # speed of light in m/s
    # findspos returns the cumulative longitudinal positions.
    # In MATLAB: Circum = findspos(THERING,1+length(THERING));
    Circum = at.get_s_pos(ring, -1)

    # Get the harmonic number from the first cavity cell.
    # In MATLAB: harmNum = THERING{icav(1)}.HarmNumber;
    harmNum = ring[icav[0]].HarmNumber
    frf = cspeed / Circum * harmNum

    # Update the HarmNumber and Frequency fields for the cavity cells.
    set_cells(ring, 'HarmNumber', harmNum, icav)
    set_cells(ring, 'Frequency', frf, icav)

    # Define initial conditions: [x, x', y, y', dpp, ct]
    # MATLAB: X0l = [pos(1), 0, pos(2), 0, dpp, 0]';
    z = np.array([pos[0], 0.0, pos[1], 0.0, dpp, 0.0])

    # Track the particle for Nturn turns.
    # MATLAB: [Rfin, loss] = ringpass(THERING, X0l, Nturn);
    tracking_result = at.lattice_pass(ring, z, Nturn, keep_lattice=False)[:, 0, 0, :]

    # Determine the number of turns survived.
    # In MATLAB, non-NaN indices in the first coordinate of Rfin are found.
    non_nan_indices = np.where(~np.isnan(tracking_result[0]))[0]
    if non_nan_indices.size > 0:
        NturnSurvived = non_nan_indices[-1] + 1
    else:
        NturnSurvived = 0

    return NturnSurvived


def track_MA(ring, dpp_index, Nturn, pos):
    """
    Simulates tracking for momentum aperture (MA) evaluation.
    
    Parameters
    ----------
    ring : list
        The accelerator lattice as a list of element dicts/objects.
    dpp_index : array-like of int
        For each momentum slice, the index in the ring where that slice should start.
    Nturn : int
        Number of turns to track.
    pos : tuple (i_slice, dpp)
        i_slice: index into dpp_index (will be cast to int)
        dpp: the momentum deviation for this slice.
    
    Returns
    -------
    NturnSurvived : int
        How many turns the particle survived before being lost.
    """
    # 1) Ensure ring is 1D
    # (MATLAB did a transpose if it had size>1, but in Python we assume a flat list)
    
    # 2) Set up Bending elements to Symplectic 4th-order radiation pass
    ib = find_cells(ring, 'BendingAngle')
    # get all non-zero bending angles
    ba = np.array([ring[i].BendingAngle for i in ib])
    ib = [i for i, angle in zip(ib, ba) if angle != 0]
    set_cells(ring, 'PassMethod', 'BndMPoleSymplectic4RadPass', ib)
    
    # 3) Distribute RF voltage evenly over all cavities
    icav = find_cells(ring, 'Frequency')
    Vrf = 6.0e6  # 6 MV
    set_cells(ring, 'Voltage', Vrf / len(icav), icav)
    
    # 4) Adjust TimeLag to keep synchronous phase correct
    try:
        # R0 = [x, x', y, y', dE/E, ct] at closed orbit
        R0, _ = at.find_orbit6(ring)
        tlag_orig = getattr(ring[icav[0]], 'TimeLag', None)
        if tlag_orig is not None:
            new_tlag = tlag_orig - R0[5]
            set_cells(ring, 'TimeLag', new_tlag, icav)
    except Exception:
        # if no TimeLag field or orbit failed, just skip
        pass
    
    # 5) Rotate the ring so that this momentum slice starts at the right place
    ii = int(pos[0]) - 1
    R0_ring = deepcopy(ring)
    cutoff = dpp_index[ii] - 1
    ring = list(R0_ring[cutoff:]) + list(R0_ring[:cutoff])
    
    # 6) Build the 6D initial vector: [x, x', y, y', dpp, ct]
    z0 = np.array([0.0, 0.0, 0.0, 0.0, pos[1], 0.0])
    
    # 7) Do the tracking
    # returns array shape (6, nturns, 1, 1) or similar — we slice out the first particle
    res = at.lattice_pass(ring, z0, Nturn, keep_lattice=False)[:, 0, 0, :]
    
    # 8) Find how many turns survived (first coordinate not NaN)
    ok = np.where(~np.isnan(res[0]))[0]
    if ok.size:
        return ok[-1] + 1
    else:
        return 0


def convert_element(el):
    # --- Robust field extraction helpers ---
    def get_str(field):
        try:
            return str(el[field][0])
        except Exception:
            return ''

    def get_float(field, default=0.0):
        try:
            val = el[field]
            # Safely extract scalar from nested arrays like [[1.23]]
            return float(np.array(val).squeeze())
        except Exception:
            return default

    def get_array(field, default=None):
        if default is None:
            default = np.zeros(1)
        try:
            arr = el[field][0]
            return np.array(arr, dtype=np.float64).flatten()
        except Exception:
            return np.array(default, dtype=np.float64)

    # --- Basic fields ---
    fam_name = get_str('FamName') or 'UNKNOWN'
    length = get_float('Length', 0.0)
    pass_method = get_str('PassMethod') or 'IdentityPass'
    mad_type = get_str('MADType') or fam_name  # fallback to FamName

    # --- Dispatcher by MADType ---
    if mad_type == 'DRIF':
        return Drift(fam_name, length, pass_method=pass_method)

    elif mad_type in ['MARK', 'MA', 'MID']:
        return Marker(fam_name, pass_method=pass_method)

    elif mad_type == 'QUAD':
        k1 = get_float('K', 0.0)
        polynom_b = get_array('PolynomB')
        polynom_a = get_array('PolynomA')
        max_order = int(get_float('MaxOrder', len(polynom_b) - 1))
        num_int_steps = int(get_float('NumIntSteps', 10))

        return Quadrupole(
            fam_name, length, k=k1,
            PolynomB=polynom_b,
            PolynomA=polynom_a,
            MaxOrder=max_order,
            NumIntSteps=num_int_steps,
            pass_method=pass_method
        )

    elif mad_type == 'SEXT':
        polynom_b = get_array('PolynomB')
        polynom_a = get_array('PolynomA')
        k2 = polynom_b[2] if len(polynom_b) > 2 else 0.0
        max_order = int(get_float('MaxOrder', len(polynom_b) - 1))
        num_int_steps = int(get_float('NumIntSteps', 10))

        return Sextupole(
            fam_name, length, h=k2,
            PolynomB=polynom_b,
            PolynomA=polynom_a,
            MaxOrder=max_order,
            NumIntSteps=num_int_steps,
            pass_method=pass_method
        )

    elif mad_type == 'OCTU':
        polynom_b = get_array('PolynomB')
        polynom_a = get_array('PolynomA')
        max_order = int(get_float('MaxOrder', len(polynom_b) - 1))
        num_int_steps = int(get_float('NumIntSteps', 10))

        return Octupole(
            fam_name, length,
            poly_b=polynom_b,
            poly_a=polynom_a,
            MaxOrder=max_order,
            NumIntSteps=num_int_steps,
            pass_method=pass_method
        )

    elif mad_type == 'MULT':
        polynom_b = get_array('PolynomB')
        polynom_a = get_array('PolynomA')
        max_order = int(get_float('MaxOrder', len(polynom_b) - 1))
        num_int_steps = int(get_float('NumIntSteps', 10))

        return Multipole(
            fam_name, length,
            poly_b=polynom_b,
            poly_a=polynom_a,
            MaxOrder=max_order,
            NumIntSteps=num_int_steps,
            pass_method=pass_method
        )

    elif mad_type == 'SBEN':
        angle = get_float('BendingAngle', 0.0)
        k = get_float('K', 0.0)
        e1 = get_float('EntranceAngle', 0.0)
        e2 = get_float('ExitAngle', 0.0)
        polynom_b = get_array('PolynomB')
        polynom_a = get_array('PolynomA')
        max_order = int(get_float('MaxOrder', len(polynom_b) - 1))
        num_int_steps = int(get_float('NumIntSteps', 10))
        energy = get_float('Energy', 0.0)

        return Dipole(
            fam_name, length,
            bending_angle=angle,
            k=k,
            EntranceAngle=e1,
            ExitAngle=e2,
            PolynomB=polynom_b,
            PolynomA=polynom_a,
            MaxOrder=max_order,
            NumIntSteps=num_int_steps,
            energy=energy,
            pass_method=pass_method
        )

    elif mad_type == 'RFCA':
        voltage = get_float('Voltage', 0.0)
        frequency = get_float('Frequency', 0.0)
        harmon = int(get_float('HarmNumber', 1))
        energy = get_float('Energy', 0.0)
        phase_lag = get_float('PhaseLag', 0.0)
        omega = 2 * np.pi * frequency
        time_lag = phase_lag / omega if omega != 0 else 0.0

        return RFCavity(
            fam_name, length,
            voltage=voltage,
            frequency=frequency,
            harmonic_number=harmon,
            energy=energy,
            TimeLag=time_lag,
            pass_method=pass_method
        )

    # --- Fallback ---
    print(f"Unknown MADType '{mad_type}' — defaulting to Marker")
    return Marker(fam_name, pass_method=pass_method)


def convert_ring(filename):

    data = loadmat(filename)
    thering = data['THERING']
    
    seed_format = len(thering) == 1
    if seed_format:
        thering = thering[0]
        pyat_ring = [convert_element(el[0, 0]) for el in thering]
    else:
        pyat_ring = [convert_element(el[0][0, 0]) for el in thering]
    thering_py = Lattice(pyat_ring, name='THERING')
    
    return thering_py


def make_config_ring(vrange4D, S0, v):

    def x_to_K2(x):
        # x: (n, 4)

        p = vrange4D[:, 0] + (vrange4D[:, 1] - vrange4D[:, 0]) * x
        K2 = S0 + v[:, 2:] @ p.T

        return K2.T


    def config_ring(ring, groups, x):
        # groups: 1d array of group
        # x: (n, 4)

        if len(x.shape) == 1:
            _x = x.reshape(1, -1)
        else:
            _x = x

        K2 = x_to_K2(_x)

        ring_list = []
        for _K2 in K2:
            _ring = deepcopy(ring)
            for idx, group in enumerate(groups):
                set_param_group(_ring, group, _K2[idx])
            ring_list.append(_ring)

        if len(ring_list) > 1:
            return ring_list
        else:
            return ring_list[0]

    return config_ring


def Y_to_qar_shape(Y, nq=None, nrays=19, npts_per_ray=72):
    if nq is None:
        nq = int(np.ceil(Y.shape[0] / nrays / npts_per_ray))

    Y_qar = np.reshape(Y, [nq, nrays, npts_per_ray], order='F')

    return Y_qar


def Y_to_train_shape(Y):
    nq, nrays, npts_per_ray = Y.shape
    Y_train = np.reshape(Y, nq * nrays * npts_per_ray, order='F')

    return Y_train


def prep_data(a: float = 5,
              npts_per_ray: int = 72,
              x0_min: float = 0.004,
              x0_max: float = 0.025,
              y0_max: float = 0.012):

    # 1) generate the 19 ray‐angles (originally from gen_ray_angles)
    x1 = np.arange(10)
    theta1 = (10 - a) * x1 + (a / 9) * x1**2
    x2 = np.arange(1, 10)
    theta2 = 90 + (10 + a) * x2 - (a / 9) * x2**2
    theta = np.concatenate([theta1, theta2]) * np.pi / 180  # shape (19,)

    # 2) radial distances along each ray
    radl = x0_min + np.arange(npts_per_ray) / npts_per_ray * (x0_max - x0_min)
    # make a (19, 72) grid of (r,θ)
    rr, aa = np.meshgrid(radl, theta)  # rr, aa both shape (19, 72)

    # 3) physical offsets in x,y
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa) * (y0_max / x0_max)

    rr = np.sqrt(xx ** 2 + yy ** 2)

    setup = np.load('setup.npz')

    angles = setup['angles']
    spos = setup['spos']
    momentum = setup['momentum']

    return angles, rr, spos, momentum


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


def calc_obj(daX, daY, obj_scaled=None):
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


def calc_maPM(turns, momentum, spos, ma_thresh=0.5, method=0):
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
            maP_pred[k, jj] = momentum[idx_P + mid]
            maM_pred[k, jj] = momentum[idx_M]

    return (maP_pred, maM_pred)


def calc_ma(spos, maP, maM, obj_scaled=None):
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


def make_calc_DAMA():

    angles, rr, spos, momentum = prep_data()

    def calc_DA(Y, threshold=1, method=1):
        YY = Y_to_qar_shape(Y)
        daX, daY = calc_daXY(YY, rr, angles, da_thresh=threshold, method=method)
        obj = calc_obj(daX, daY)

        return obj

    def calc_MA(Y, threshold=1, method=2):
        YY = Y_to_qar_shape(Y, nrays=21, npts_per_ray=49)
        maP, maM = calc_maPM(YY, momentum, spos, ma_thresh=threshold, method=method)
        obj = calc_ma(spos, maP, maM)

        return obj

    return calc_DA, calc_MA


def make_fn_oracle(SEXT, DAMAPS, MAMAPS):

    theta, rr, spos, momentum = prep_data()

    # X: (n, 6 or 7) in raw -- exact black box input
    # Y: (n, 10) in raw -- exact black box output
    def fn_DA_oracle(X):

        conf = X[:, :4]
        x = X[:, 4]
        y = X[:, 5]

        angle = np.arctan2(y, x)
        radius = np.sqrt(x ** 2 + y ** 2).reshape(-1, 1)

        indices_conf = find_row_indices(conf, SEXT)
        indices_angle = find_element_indices(angle, theta)
        indices_radius = np.argmin(np.abs(rr[indices_angle] - radius), axis=1)
        # seed = np.random.choice(10)

        Y = DAMAPS[indices_conf, :, indices_angle, indices_radius]

        return Y
        # return Y.reshape(-1, 1)


    # X: (n, 6 or 7) in raw -- exact black box input
    # Y: (n, 10) in raw -- exact black box output
    def fn_MA_oracle(X):

        conf = X[:, :4]
        m = X[:, 5]

        indices_conf = find_row_indices(conf, SEXT)
        indices_spos = (X[:, 4] - 1).astype(int)
        indices_momentum = find_element_indices(m, momentum)
        # seed = np.random.choice(10)

        Y = MAMAPS[indices_conf, :, indices_spos, indices_momentum]

        return Y

    return fn_DA_oracle, fn_MA_oracle


# Load the variables
data = loadmat('data_setup_H6BA_10b_6var.mat')
SextPara = data['SextPara'][0]
g_dpp_index = data['g_dpp_index']

data = loadmat('init_config.mat')
vrange4D = data['vrange4D']
S0 = data['S0']
v = data['v']

# Set tracking params
Nturn = 512 * 2

# Load the rings
rings = []
with open("rings/ring0.pkl", "rb") as f:
    rings.append(pickle.load(f))
for i in range(1, 26):
    with open(f"rings/ring_s{i}.pkl", "rb") as f:
        rings.append(pickle.load(f))

config_ring = make_config_ring(vrange4D, S0, v)


def eval_DA(x):
    
    thering = rings[int(x[6])]
    
    _ring = config_ring(thering, SextPara, x[None, :4])
    survived_turns = track_DA(_ring, 0, Nturn, x[4:6])
    y = survived_turns

    return y


def eval_MA(x):
    
    thering = rings[int(x[6])]
    
    _ring = config_ring(thering, SextPara, x[None, :4])
    survived_turns = track_MA(_ring, g_dpp_index[0], Nturn, x[4:6])
    y = survived_turns

    return y


def evaluate_DA(X):
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(eval_DA, X))
    Y = np.array(results).reshape(-1, 1)
    
    return Y


def evaluate_MA(X):
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(eval_MA, X))
    Y = np.array(results).reshape(-1, 1)

    return Y
