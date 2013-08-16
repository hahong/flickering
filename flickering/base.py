"""Base functions for flickering analysis"""
import numpy as np
import os
import glob


# -- some helper functions
def grouper(iterable, n, offset=0):
    """Create chunks of size n."""
    l = len(iterable)
    idxs = range(offset, l, n)
    idxs.append(l)
    for ib, ie in zip(idxs[:-1], idxs[1:]):
        if ie - ib != n:
            continue
        yield iterable[ib: ie]


# -- data IO functions
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
        return   # nothing to read

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
    do_twoway=True,
    scale_num=32, scale_begin_fac=4):
    """Perform DFA analysis.

Input
-----
signal: signal
scales: a vector of scales
deg: detrend polynomial order
full: if True, some diagnostic variables will be returned as well.

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
