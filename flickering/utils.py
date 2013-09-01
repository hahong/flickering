"""Utility functions for flickering analysis"""
from . import base
from scipy import io
import numpy as np
import cPickle as pk
import os


# -- valid list of statistics
REGISTRY_STATS_FAST = {
        'mean': base.smean,
        'smean': base.smean,   # alias
        'std': base.sstd,
        'sstd': base.sstd,     # alias
        'cv': base.cv,
    }

REGISTRY_STATS_SLOW = {
        'hurst': base.hurst,
        'dfa': base.dfa,
        'mse': base.mse,
        'fftslope': base.fft_slope,
        'fft_slope': base.fft_slope,   # alias
    }

# -- templates
ANALYSIS_CFGS_TEMPLATE_FAST = [
        {'stat': 'mean'},
        {'stat': 'std'},
        {'stat': 'cv'},
    ]

ANALYSIS_CFGS_TEMPLATE_FULL = [
        {'stat': 'mean'},
        {'stat': 'std'},
        {'stat': 'cv'},
        {'stat': 'hurst'},
        {'stat': 'dfa'},
        {'stat': 'mse'},
        {'stat': 'fftslope'},
    ]


# -- utility functions
def write_csv_2d(path, arr):
    assert arr.ndim == 2
    fp = open(path, 'wt')
    for r in arr:
        s = ','.join('%1.18e' % c for c in r)
        fp.write(s + '\n')
    fp.close()


def write_res_multiple(prefix, mdict, npz=True, mat=True, pkl=False):
    """Write multiple results."""
    if npz:
        np.savez(prefix + '.npz', **mdict)
    if mat:
        io.savemat(prefix + '.mat', mdict, oned_as='row')
    if pkl:
        pk.dump(mdict, open(prefix + '.pkl', 'wb'))


def write_res_single(prefix, rarr, txt1d=True, txt2d=True, csv2d=True):
    """Write a single result."""
    if txt1d:
        np.savetxt(prefix + '.1d.txt', np.ravel(rarr))
    if txt2d:
        np.savetxt(prefix + '.2d.txt', rarr)
    if csv2d:
        write_csv_2d(prefix + '.csv', rarr)


def run_analysis(frames, cfgs, outprefix, n_jobs=1, verbose=0, savmulti=True):
    """One-shot function to run analysis"""
    res = {}
    res_raw = {}

    for cfg in cfgs:
        stat = cfg['stat']
        comp_kw = cfg.get('comp_kw', {})
        wrisg_kw = cfg.get('wrisg_kw', {})
        israw = comp_kw.get('raw', False)

        if verbose > 0:
            print '-->', stat
            print cfg

        if stat in REGISTRY_STATS_FAST:
            func = REGISTRY_STATS_FAST[stat]
            r = func(frames, **comp_kw)
        elif stat in REGISTRY_STATS_SLOW:
            func = REGISTRY_STATS_SLOW[stat]
            r = base.compute_stats(frames, func, n_jobs=n_jobs,
                    verbose=verbose, **comp_kw)
        else:
            raise ValueError('Unknown "stat"')

        if not israw:
            res[stat] = r
            write_res_single(outprefix + '.' + stat, r, **wrisg_kw)
        else:
            res_raw[stat] = r

    if savmulti:
        write_res_multiple(outprefix + '.all', res, pkl=False)
        if len(res_raw) > 0:
            res.update(res_raw)
            write_res_multiple(outprefix + '.all', res, npz=False,
                    mat=False, pkl=True)


# -- CLI functions
def parse_opts(opts0):
    """Parse the options in the command line.  This somewhat
    archaic function mainly exists for backward-compatability."""
    opts = {}
    # parse the stuff in "opts"
    for opt in opts0:
        parsed = opt.split('=')
        key = parsed[0].strip()
        if len(parsed) > 1:
            # OLD: cmd = parsed[1].strip()
            cmd = '='.join(parsed[1:]).strip()
        else:
            cmd = ''
        opts[key] = cmd

    return opts


def parse_opts2(tokens, optpx='--', argparam=False):
    """A newer option parser."""
    opts0 = []
    args = []
    n = len(optpx)

    for token in tokens:
        if token.startswith(optpx):
            opts0.append(token[n:])
        else:
            if argparam:
                token = token.split('=')
            args.append(token)

    opts = parse_opts(opts0)

    return args, opts


def cli_flickering_analysis(full_argv):
    """Command line interface"""
    args, opts = parse_opts2(full_argv[1:])

    if len(args) != 2:
        usage = """Usage:
$EXEC [options] <input path> <output path prefix>

Options:
--full         Compute full statistics including DFA, MSE, and FFT.
--n_jobs=#     Set the number of worker threads (default=1).
--verbose=#    Set the verbosity (default=1).
--featnorm     Do feature normalization first.
"""
        usage = usage.replace('$EXEC', os.path.basename(full_argv[0]))
        print usage
        return 1

    # -- do work
    inp, outp = args
    full = False
    featnorm = False
    n_jobs = 1
    verbose = 1

    if 'full' in opts:
        full = True
        print '* full stats'
    if 'featnorm' in opts:
        featnorm = True
        print '* feature normalization'
    if 'n_jobs' in opts:
        n_jobs = int(opts['n_jobs'])
        print '* n_jobs =', n_jobs
    if 'verbose' in opts:
        verbose = int(opts['verbose'])
        print '* verbose =', verbose

    frames = base.load_data(inp)
    if featnorm:
        frames = base.feat_normalize(frames)
    print '* data shape =', frames.shape

    if full:
        cfgs = ANALYSIS_CFGS_TEMPLATE_FULL
    else:
        cfgs = ANALYSIS_CFGS_TEMPLATE_FAST

    run_analysis(frames, cfgs, outp, verbose=verbose, n_jobs=n_jobs)
    return 0


# -- Some protection against parallel running..
if __name__ == '__main__':
    pass   # do nothing
