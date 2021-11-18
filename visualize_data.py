import os
import argparse
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIG_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title

def read_data(paths):
    jet_dfs = []
    pf_arrays = []
    for path in paths:
        f = uproot.open(path)['Jets']
        jet_df = f.arrays(['pt', 'pt_gen', 'eta', 'eta_gen', 'phi', 'parton_flavor', 'hadron_flavor', 'pt_full_corr'], library='pd')
        flavour = jet_df.hadron_flavor.where(jet_df.hadron_flavor != 0, other=np.abs(jet_df.parton_flavor))
        jet_df = jet_df.drop(columns=['parton_flavor', 'hadron_flavor'])
        jet_df['flavour'] = flavour.map({1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 21: 'g', 0: 'unknown'})
       
        jet_dfs.append(jet_df)

        pf = f.arrays(['ch_pt', 'ne_pt', 'ch_eta', 'ne_eta', 'ch_phi', 'ne_phi'])
        pf_pt = ak.concatenate((pf.ch_pt, pf.ne_pt), axis=1)
        pf_eta = ak.concatenate((pf.ch_eta, pf.ne_eta), axis=1)
        pf_phi = ak.concatenate((pf.ch_phi, pf.ne_phi), axis=1)

        pf_arrays.append(ak.zip({'pt': pf_pt, 'eta': pf_eta, 'phi': pf_phi}))

    jet_df = pd.concat(jet_dfs, axis=0)
    pf_array = ak.concatenate(pf_arrays, axis=0)

    return jet_df, pf_array


def sinhspace(start, stop, step):
    sinh_array = np.sinh(np.linspace(start, np.pi, step))
    return sinh_array / sinh_array.max() * stop


def plot_spectrum(jet_df, outdir):
    pt_bins = sinhspace(0, 3000, 50)
    eta_bins = np.linspace(0, jet_df.eta_gen.abs().max(), 30)

    H, _, _ = np.histogram2d(jet_df.pt_gen, jet_df.eta_gen.abs(), bins=(pt_bins, eta_bins))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot()

    plt.imshow(np.flipud(H.T), aspect='auto', norm=mpl.colors.LogNorm(vmin=1, vmax=H.max()))
    plt.xlabel('$p^\\mathrm{{gen}}_{T}$')
    plt.ylabel('$|\\eta^\\mathrm{{gen}}|$')
    plt.colorbar()

    xlim = plt.gca().get_xlim()
    xrange = np.linspace(xlim[0], xlim[1], 50)
    pt_ticks = [0, 30, 100, 300, 1000, 3000]
    plt.xticks(ticks=np.interp(pt_ticks, pt_bins, xrange), labels=pt_ticks)

    ylim = plt.gca().get_ylim()
    yrange = np.linspace(ylim[0], ylim[1], 30)
    eta_ticks = [0, 1, 2, 3, 4, 5]
    plt.yticks(ticks=np.interp(eta_ticks, eta_bins, yrange), labels=eta_ticks)

    minor_ticks = list(range(0, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 5000, 1000))
    ax.set_xticks(ticks=np.interp(minor_ticks, pt_bins, xrange), minor=True)
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'spectrum.{ext}'))
    plt.close(fig)


def plot_flavour_bars(jet_df, outdir):
    plt.figure(figsize=(8, 6.5))
    jet_df.flavour.value_counts(sort=False).loc[['u', 'd', 's', 'c', 'b', 'g', 'unknown']].plot.bar(ylabel='Number of jets')
    plt.xticks(rotation=0)
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'flavors.{ext}'))
    plt.close()


def get_hist_coords(data, nbins):
    hist, bin_edges = np.histogram(data, bins=nbins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers, hist


def plot_target(target, outdir):
    x, y = get_hist_coords(target, 1000)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot()
    ax.step(x, y / y.sum(), linewidth=2)
    ax.set_ylim(0, None)
    ax.set_xlabel('$\log(p_T^\mathrm{gen} / p_T^\mathrm{reco})$')
    ax.set_ylabel('Fraction of jets/bin')
    ax.set_xlim((-1.5, 1.5))
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'target.{ext}'))
    plt.close(fig)


def qgl_step(sample, feature, label, nbins, outdir):
    if feature == 'mult':
        quark_y = sample[sample.flavour.isin({'u', 'd', 's'})]['mult'].value_counts(sort=False)
        quark_x = quark_y.index
        gluon_y = sample[sample.flavour.isin({'g'})]['mult'].value_counts(sort=False)
        gluon_x = gluon_y.index
    else:
        quark_x, quark_y = get_hist_coords(sample[sample.flavour.isin({'u', 'd', 's'})][feature], nbins)
        gluon_x, gluon_y = get_hist_coords(sample[sample.flavour.isin({'g'})][feature], nbins)
    
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot()
    ax.step(quark_x, quark_y / quark_y.sum(), linewidth=2, label='quark')
    ax.step(gluon_x, gluon_y / gluon_y.sum(), linewidth=2, label='gluon')
    ax.set_ylim(0, None)
    ax.set_xlabel(label)
    ax.set_ylabel('Fraction of jets/bin')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.text(
        1., 1.002,
        r'${:g} < p_\mathrm{{T}}^\mathrm{{gen}} < {:g}$ GeV, '
        r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(80, 100, 0, 1.3),
        ha='right', va='bottom', transform=ax.transAxes, fontsize=SMALL_SIZE
        )
    ax.legend()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'{feature}.{ext}'))
    plt.close(fig)


def plot_qgl(jet_df, pf, outdir):
    deta = pf.eta - jet_df.eta
    dphi = pf.phi - jet_df.phi
    weight = pf.pt**2

    sum_weight = ak.sum(weight, axis=1)
    ave_deta = ak.sum(deta * weight, axis=1) / sum_weight
    ave_dphi = ak.sum(dphi * weight, axis=1) / sum_weight
    ave_deta2 = ak.sum(deta**2 * weight, axis=1) / sum_weight
    ave_dphi2 = ak.sum(dphi**2 * weight, axis=1) / sum_weight
    sum_detadphi = ak.sum(deta * dphi * weight, axis=1)

    a = ave_deta2 - ave_deta * ave_deta
    b = ave_dphi2 - ave_dphi * ave_dphi
    c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi)

    delta = np.sqrt(np.abs((a - b) * (a - b) + 4 * c * c))

    jet_df['mult'] = np.array(ak.num(pf.pt[pf.pt > 1]))
    jet_df['ptD'] = np.where(sum_weight > 0, np.sqrt(sum_weight) / ak.sum(pf.pt, axis=1), 0)
    jet_df['axis2'] = np.where(a + b - delta > 0, np.sqrt(0.5 * (a + b - delta)), 0)

    sample = jet_df[
        (jet_df.pt_gen >= 80) & (jet_df.pt_gen < 100)
        & (np.abs(jet_df.eta_gen) < 1.3)
    ]
    nbins = sample.mult.nunique()

    qgl_step(sample, 'mult', 'Multiplicity', nbins, outdir)
    qgl_step(sample, 'ptD', '$p_TD$', nbins, outdir)
    qgl_step(sample, 'axis2', '$\sigma_2$', nbins, outdir)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-i', '--indir', required=True, help='Data directory')
    arg_parser.add_argument('-o', '--outdir', required=True, help='Where to store plots')
    args = arg_parser.parse_args()

    try:
        for ext in ['png', 'pdf']:
            os.makedirs(os.path.join(args.outdir, ext))
    except FileExistsError:
        pass
    
    jet_df, pf_array = read_data(glob(os.path.join(args.indir, '*.root')))
    
    plot_spectrum(jet_df, args.outdir)

    plot_target(np.log(jet_df.pt_gen / jet_df.pt), args.outdir)

    plot_qgl(jet_df, pf_array, args.outdir)

    plot_flavour_bars(jet_df, args.outdir)