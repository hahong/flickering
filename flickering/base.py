"""Base functions for flickering analysis"""
import numpy as np
import os
import glob
from joblib import Parallel, delayed


# -- Some helper functions
def grouper(iterable, n, offset=0):
    """Create chunks of size n."""
    l = len(iterable)
    idxs = range(offset, l, n)
    idxs.append(l)
    for ib, ie in zip(idxs[:-1], idxs[1:]):
        if ie - ib != n:
            continue
        yield iterable[ib: ie]


# -- Data IO functions
def load_once_csv(f, sep=','):
    """Load a single frame in a csv file."""
    lines = [e.strip().split(sep) for e in open(f).readlines()]
    M = []
    for line in lines:
        # ignore the line if an empty entry exists
        if '' in line:
            continue
        # otherwise process the line
        M.append([float(e) for e in line])
    M = np.array(M)
    return M


def load_data(path, dattype='csvdir', patt=None, kw_loader={}, flist=None):
    """Load flickering data (movie)."""
    # prep...
    if dattype == 'csvdir':
        patt_ = '*.csv'
        backend = load_once_csv
    else:
        raise ValueError('Unrecognized "dattype"')

    if patt is None:
        patt = patt_

    # get file lists and data
    if flist is None:
        flist = sorted(glob.glob(path + os.path.sep + patt))

    if len(flist) == 0:
        raise ValueError('No files to read')

    frames = []
    for f in flist:
        frame = backend(f)
        frames.append(frame, **kw_loader)

    # check if all frames have the same x y dimension...
    if all([e.shape == frames[0].shape for e in frames]):
        frames = np.array(frames)
    else:
        print 'Warning: some frame(s) have different dimensions'

    return frames


# -- Main analysis functions
def hurst(signal, ns=None, full=False, n_begin=10, n_num=128):
    """Compute Hurst exponent.

Input
-----
signal: signal
ns: a list of n (point to make an evaluation)
full: if True, some diagnostic variables will be returned as well.

Reference
---------
http://cds.cern.ch/record/595648/files/0212029.pdf
# the definition of m and E[R/S] is little unclear:
http://en.wikipedia.org/wiki/Hurst_exponent
"""
    if len(signal.shape) != 1:
        raise ValueError('"signal" must be 1D.')

    if ns is None:
        ns = np.logspace(np.log10(n_begin), np.log10(len(signal)), num=n_num)
    elif ns == 'full':
        ns = np.arange(2, len(signal) + 1)

    # mean
    # incorrect: m = np.mean(signal)
    m = np.array([np.mean(signal[:l]) for l in xrange(1, len(signal) + 1)])
    # cumulative deviations
    Z = np.cumsum(signal - m)
    # ranges
    R = np.array([np.max(Z[:n]) - np.min(Z[:n]) for n in ns])
    # stddvs (TODO: but shouldn't we use sample stddv?)
    S = np.array([np.mean((signal - m)[:n] ** 2) for n in ns])
    S = np.sqrt(S)
    # expected values of the rescaled range, R/S
    # incorrect: E = [np.mean(r/s) for r, s in zip(R, S)]
    E = R / S

    p = np.polyfit(np.log(ns), np.log(E), 1)
    # Hurst exponent
    H = p[0]

    if not full:
        return H

    return H, E, R, S, ns


def get_detrended_var(y_chunk, deg=1):
    """Get the detrended variance."""
    x = np.arange(len(y_chunk))
    p = np.polyfit(x, y_chunk, deg)
    yn_chunk = np.polyval(p, x)
    return np.var(y_chunk - yn_chunk, ddof=1)


def dfa(signal, scales=None, deg=1, full=False,
    do_twoway=True, scale_num=32, scale_begin_fac=4):
    """Perform DFA analysis.

Input
-----
signal: signal
scales: a vector of scales
deg: detrend polynomial order
full: if True, some diagnostic variables will be returned as well.
do_twoway: if True (default), 2nd calculation will be done in backward.

Reference
---------
www.uni-giessen.de/physik/theorie/theorie3/publications/PhysicaA-2001-2.pdf
Peng C-K, Havlin S, Stanley HE, Goldberger AL. Chaos 1995;5:82.
http://en.wikipedia.org/wiki/Detrended_fluctuation_analysis
"""
    if len(signal.shape) != 1:
        raise ValueError('"signal" must be 1D.')

    if scales is None:
        scales = np.logspace(np.log10(deg + 1) * scale_begin_fac,
            np.log10(len(signal)),
            num=scale_num)

    # cumulative sum
    y = np.cumsum(signal - np.mean(signal))
    ns = []
    Fns = []

    for scale in scales:
        n = int(np.round(scale))    # nicer scale
        vs = []                     # variances

        # forward
        for y_chunk in grouper(y, n):
            v = get_detrended_var(y_chunk, deg=deg)
            vs.append(v)

        # do the same in the reverse direction
        if do_twoway:
            for y_chunk in grouper(y[::-1], n):
                v = get_detrended_var(y_chunk, deg=deg)
                vs.append(v)

        if len(vs) == 0:
            continue

        rms = np.sqrt(np.mean(vs))
        Fns.append(rms)
        ns.append(n)

    # get the exponent alpha
    p = np.polyfit(np.log(ns), np.log(Fns), 1)
    alpha = p[0]

    if not full:
        return alpha
    return alpha, ns, Fns


def cv(frames):
    """Get coefficient of vaiation."""
    s = np.std(frames, axis=0, ddof=1)
    m = np.mean(frames, axis=0)
    return s / m


def sampen(y, M, r):
    """Compute SampEn.

Input
-----
y: input data
M: max template length
r: matching tolerance

Output
------
e: sample entropy estimates for m = 0, 1, ..., M - 1
A: number of matches for m = 1, ..., M
B: number of matches for m = 0, ..., M - 1 excluding last point


References
----------
http://www.physionet.org/physiotools/sampen/matlab/1.1-1/
http://www.physionet.org/physiotools/mse/tutorial/tutorial.pdf
http://ajpheart.physiology.org/content/278/6/H2039
http://physionet.org/physiotools/mse/mse.c
"""
    # XXX: Since this is a dead copy of the matlab code in the reference
    # the index starts from 1...
    n = len(y)
    lastrun = np.zeros(n + 1, 'int')
    run = np.zeros(n + 1, 'int')
    A = np.zeros(M + 1, 'int')
    B = np.zeros(M + 1, 'int')
    p = np.zeros(M + 1, 'int')
    e = np.zeros(M + 1, 'int')

    for i in xrange(1, n):
        nj = n - i
        y1 = y[i - 1]   # y is 0-based

        for jj in xrange(1, nj + 1):
            j = jj + i
            if np.abs(y[j - 1] - y1) < r:  # y is 0-based
                run[jj] = lastrun[jj] + 1
                M1 = min(M, run[jj])
                for m in xrange(1, M1 + 1):
                    A[m] += 1
                    if j < n:
                        B[m] += 1
            else:
                run[jj] = 0

        for j in xrange(1, nj + 1):
            lastrun[j] = run[j]

    N = n * (n - 1.) / 2.
    A = A[1:]
    B[0] = N
    B = B[:M]
    p = A.astype('float') / B
    e = -np.log(p)

    return e, A, B


def sampen_scale2_py(y, r):
    """Compute SampEn at m = 2 (Python version).

Input
-----
y: input data
r: matching tolerance

References
----------
Wu et al., Entropy 2013, 15, 1069-1084
"""
    l = len(y)
    Nn = 0
    Nd = 0
    for i in xrange(l - 2):
        for j in xrange(i + 1, l - 2):
            if np.abs(y[i] - y[j]) < r and np.abs(y[i + 1] - y[j + 1]) < r:
                Nn += 1

                if np.abs(y[i + 2] - y[j + 2]) < r:
                    Nd += 1

    return -np.log(float(Nd) / Nn)


try:
    # try to import fast C version
    from . import ffast as ff
    sampen_scale2 = ff.sampen_scale2_f64f64
except:
    sampen_scale2 = sampen_scale2_py


def mse(signal, r=0.15, scales=None, scale_num=6, scale_min=1):
    """Perform DFA analysis.

Input
-----
signal: signal
r: tolerance for template matching
scales: a vector of scales

Reference
---------
http://www.physionet.org/physiotools/mse/tutorial/tutorial.pdf
"""
    if len(signal.shape) != 1:
        raise ValueError('"signal" must be 1D.')

    if scales in None:
        scales = range(scale_min, scale_min + scale_num)


# -- Analysis driver functions
def compute_stats(frames, func=hurst, kw_func={}, n_jobs=1, raw=False,
        verbose=0, subsmp=None):
    """Analysis driver function.

Inputs
------
frames: frames of flickering data.  Must be in 3D of (time, x, y)
func: the function that computes the desired statistics
kw_func: keyword arguments that will be passed to the func()
subsmp: a 2D bitmap that specifies on which pixels the statistics
    will be evaluated.
n_jobs: the number of worker threads

Returns
-------
M: the 2D matrix of the statstics across pixels (x, y)
"""
    if len(frames.shape) != 3:
        raise ValueError('"frames" not in rank-3 array')

    r, c = frames.shape[1:]
    if subsmp is None:
        subsmp = np.ones((r, c))
    else:
        assert subsmp.shape == (r, c)
    subsmp = subsmp.astype('bool')

    M = np.empty((r, c))
    M[:, :] = np.nan

    R = Parallel(n_jobs, verbose=verbose)(delayed(func)(frames[:, r_, c_],
        **kw_func) for r_ in xrange(r) for c_ in xrange(c) if subsmp[r_, c_])
    if raw:
        return R

    M[subsmp] = R
    return M


# -- Some protection against parallel running..
if __name__ == '__main__':
    pass   # do nothing
