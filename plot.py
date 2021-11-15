import os
import argparse
import pickle
import itertools
import uproot
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

MARKERS = ['o', 's', 'D', '^', 'v']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def read_data(paths, predictions):
    dfs = []
    for path in paths:
        df = uproot.open(path)['Jets'].arrays(['pt', 'pt_gen', 'eta_gen', 'parton_flavor', 'hadron_flavor', 'pt_full_corr'], library='pd')
        flavour = df.hadron_flavor.where(df.hadron_flavor != 0, other=np.abs(df.parton_flavor))
        df = df.drop(columns=['parton_flavor', 'hadron_flavor'])
        df['flavour'] = flavour
       
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    df['Standard'] = df.pt_full_corr / df.pt_gen

    for key, values in predictions.items():
        corrected_pt = np.exp(values.flatten()) * df.pt
        df[key] = corrected_pt / df.pt_gen

    return df


def plot_distrs(dataframe, names, fig_dir):
    """Plot distributions of response in a few representative bins."""

    binning = np.linspace(0.5, 1.5, num=101)
    pt_bins = [(30, 40), (100, 110), (1000, 1100)]
    eta_bins = [(0., 2.5), (2.5, 5)]

    histograms = {}
    for name in names:
        histograms[name] = {}
    for (ipt, pt_bin), (ieta, eta_bin) in itertools.product(
        enumerate(pt_bins), enumerate(eta_bins)
    ):
        df_bin = dataframe[
            (dataframe.pt_gen >= pt_bin[0]) & (dataframe.pt_gen < pt_bin[1])
            & (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
        ]
        for label, selection in [
            ('uds', (df_bin.flavour <= 3) & (df_bin.flavour != 0)),
            ('b', df_bin.flavour == 5),
            ('g', df_bin.flavour == 21)
        ]:
            for name in names:
                h, _ = np.histogram(df_bin[name][selection], bins=binning)
                histograms[name][ipt, ieta, label] = h

    for ipt, ieta, flavour in itertools.product(
        range(len(pt_bins)), range(len(eta_bins)), ['uds', 'b', 'g']
    ):
        fig = plt.figure()
        ax = fig.add_subplot()
        for i, name in enumerate(names):
            ax.hist(
                binning[:-1], weights=histograms[name][ipt, ieta, flavour],
                bins=binning, histtype='step', label=name, color=COLORS[i])
        ax.axvline(1., ls='dashed', lw=0.8, c='gray')
        ax.margins(x=0)
        ax.set_xlabel(
            r'$p_\mathrm{T}^\mathrm{corr}\//\/p_\mathrm{T}^\mathrm{gen}$')
        ax.set_ylabel('Jets')
        ax.legend(loc='upper right')
        ax.text(
            1., 1.002,
            r'${}$, ${:g} < p_\mathrm{{T}}^\mathrm{{gen}} < {:g}$ GeV, '
            r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                flavour, pt_bins[ipt][0], pt_bins[ipt][1],
                eta_bins[ieta][0], eta_bins[ieta][1]
            ),
            ha='right', va='bottom', transform=ax.transAxes
        )
        ax.tick_params(
            axis='both', which='both', direction='in', 
            bottom=True, top=True, left=True, right=True
        )

        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_dir, ext, f'{flavour}_pt{ipt + 1}_eta{ieta + 1}.{ext}'))
        plt.close(fig)


def bootstrap_median(x, num=30):
    """Compute errors on median with bootstrapping."""

    if len(x) == 0:
        return np.nan

    medians = []
    for _ in range(num):
        x_resampled = np.random.choice(x, len(x))
        medians.append(np.median(x_resampled))
    return np.std(medians)


def compare_flavours(dataframe, names, fig_dir):
    """Plot median response as a function of jet flavour."""
    
    pt_cut = 30
    for ieta, eta_bin in enumerate([(0, 2.5), (2.5, 5)], start=1):
        df_pteta = dataframe[
            (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
            & (dataframe.pt_gen > pt_cut)
        ]
        median, median_error = {}, {}
        for name in names:
            median[name], median_error[name] = [], []
        flavours = [('u', {1}), ('d', {2}), ('s', {3}), ('c', {4}), ('b', {5}), ('g', {21})]
        for _, pdg_ids in flavours:
            df = df_pteta[df_pteta.flavour.isin(pdg_ids)]
            for name in names:
                median[name].append(df[name].median())
                median_error[name].append(bootstrap_median(df[name]))

        fig = plt.figure()
        ax = fig.add_subplot()
        for i, name in enumerate(names):
            ax.errorbar(
                np.arange(len(flavours)) - 0.04, median[name], yerr=median_error[name],
                color=COLORS[i], marker=MARKERS[i], ms=3, lw=0, elinewidth=0.8, label=name
            )
        ax.set_xlim(-0.5, len(flavours) - 0.5)
        ax.axhline(1, ls='dashed', lw=0.8, c='gray')
        ax.set_xticks(np.arange(len(flavours)))
        ax.set_xticklabels([f[0] for f in flavours])
        ax.legend()
        ax.set_ylabel('Median response')
        ax.text(
            1., 1.002,
            r'$p_\mathrm{{T}}^\mathrm{{gen}} > {:g}$ GeV, '
            r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                pt_cut, eta_bin[0], eta_bin[1]
            ),
            ha='right', va='bottom', transform=ax.transAxes
        )
        ax.tick_params(
            axis='both', which='both', direction='in', 
            bottom=True, top=True, left=True, right=True
        )
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(fig_dir, ext, f'eta{ieta}.{ext}'))
        plt.close(fig)


def plot_median_response(outdir, flavour_label, bins, bin_centers, eta_bin, ieta, names):
    """Plot median response as a function of pt."""

    median, median_error = {}, {}
    for name in names:
        median[name] = bins[name].median().to_numpy()
        median_error[name] = np.empty_like(median[name])
        for i, (_, df) in enumerate(bins):
            median_error[name][i] = bootstrap_median(df[name].to_numpy())

    fig = plt.figure()
    
    ax = fig.add_subplot()
    
    for i, name in enumerate(names):
        ax.errorbar(
            bin_centers, median[name], yerr=median_error[name],
            color=COLORS[i], ms=3, fmt=MARKERS[i], elinewidth=0.8, label=name
        )
    ax.axhline(1, ls='dashed', c='gray', alpha=.7)
    ax.set_xlabel('$p^\\mathrm{{gen}}_{T}$')
    ax.set_ylabel('Median response')
    ax.text(
        1., 1.002,
        '{}${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(
            f'${flavour_label}$, ' if flavour_label != 'all' else '',
            eta_bin[0], eta_bin[1]
        ),
        ha='right', va='bottom', transform=ax.transAxes
    )
    ax.legend(loc='upper right')
    ax.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )
    ax.set_xscale('log')

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, ext, f'{flavour_label}_eta{ieta}.{ext}'))
    plt.close(fig)


def bootstrap_iqr(x, num=30):
    """Compute errors on IQR with bootstrapping."""

    if len(x) == 0:
        return np.nan

    iqrs = []
    for _ in range(num):
        x_resampled = np.random.choice(x, len(x))
        quantiles = np.percentile(x_resampled, [25, 75])
        iqrs.append(quantiles[1] - quantiles[0])
    return np.std(iqrs)


def compute_iqr(groups):
    """Compute IQR from series GroupBy."""
    
    q = groups.quantile([0.25, 0.75])
    iqr = q[1::2].values - q[0::2].values

    return iqr


def plot_resolution(outdir, flavour_label, bins, bin_centers, eta_bin, ieta, names):
    median, iqr, iqr_error = {}, {}, {}
    for name in names:
        median[name] = bins[name].median().to_numpy()
        iqr[name] = compute_iqr(bins[name])
        iqr_error[name] = np.empty_like(iqr[name])
        for i, (_, df) in enumerate(bins):
            iqr_error[name][i] = bootstrap_iqr(df[name].to_numpy())

    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(2, 1, hspace=0.02, height_ratios=[4, 1])
    axes_upper = fig.add_subplot(gs[0, 0])
    axes_lower = fig.add_subplot(gs[1, 0])

    for i, name in enumerate(names):
        axes_upper.errorbar(
            bin_centers, iqr[name] / median[name], yerr=iqr_error[name] / median[name],
            color=COLORS[i], ms=3, marker=MARKERS[i], lw=0, elinewidth=0.8, label=name
        )
        if name != 'Standard':
            axes_lower.plot(
                bin_centers, (iqr[name] / median[name]) / (iqr['Standard'] / median['Standard']),
                color=COLORS[i], ms=3, marker=MARKERS[i], lw=0
            )

    axes_upper.set_ylim(0, None)
    if eta_bin[0] == 0:
        axes_lower.set_ylim(0.85, 1.02)
    else:
        axes_lower.set_ylim(0.78, 1.02)
        axes_lower.set_yticks([0.8, 0.9, 1.0])
    for axes in [axes_upper, axes_lower]:
        axes.set_xscale('log')
        axes.set_xlim(binning[0], binning[-1])
    axes_upper.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    axes_upper.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes_upper.legend()
    axes_upper.text(
        1, 1.002,
        '{}${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(
            f'${flavour_label}$, ' if flavour_label != 'all' else '',
            eta_bin[0], eta_bin[1]
        ),
        ha='right', va='bottom', transform=axes_upper.transAxes
    )
    axes_upper.set_ylabel('IQR / median for response')
    axes_lower.set_ylabel('Ratio')
    axes_lower.set_xlabel(r'$p_\mathrm{T}^\mathrm{gen}$')
    axes_upper.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )
    axes_lower.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )
    fig.align_ylabels()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, ext, f'{flavour_label}_eta{ieta}_iqr.{ext}'))
    plt.close(fig)


def plot_median_residual(outdir, bin_centers, flavour_labels, bins, eta_bin, ieta, names):
    """Plot difference in median response between flavours as a function of pt."""

    median, median_error, difference, error = {}, {}, {}, {}
    for name in names:
        median[name], median_error[name] = {}, {}
        for i in [0, 1]:
            median[name][i] = bins[i][name].median().to_numpy()
            median_error[name][i] = np.empty_like(median[name][i])
            for j, (_, df) in enumerate(bins[i]):
                median_error[name][i][j] = bootstrap_median(df[name].to_numpy())

        difference[name] = median[name][0] - median[name][1]
        error[name] = np.sqrt(median_error[name][0] ** 2 + median_error[name][1] ** 2)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i, name in enumerate(names):
        ax.errorbar(
            bin_centers, difference[name], yerr=error[name], 
            color=COLORS[i], ms=3, fmt=MARKERS[i], elinewidth=0.8, label=name
        )
    ax.axhline(0, ls='dashed', c='gray', alpha=.7)
    ax.set_xlabel('$p^\\mathrm{{gen}}_{T}$')
    ax.set_ylabel('$R_{' + flavour_labels[0] + '}-R_{' + flavour_labels[1] + '}$')
    ax.text(
        1., 1.002,
        '${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(eta_bin[0], eta_bin[1]),
        ha='right', va='bottom', transform=ax.transAxes
    )
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(outdir, f'{flavour_labels[0]}-{flavour_labels[1]}_eta{ieta}.{ext}'))
    plt.close(fig)


def list_str(values):
    lst = values.split(',')
    return [val.strip() for val in lst]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-i', '--indirs', required=True, type=list_str, help='Training results directory')
    arg_parser.add_argument('-n', '--names', required=True, type=list_str, help='Model name')
    arg_parser.add_argument('-o', '--outdir', required=True, help='Where to store plots')
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    all_predictions = {}
    all_test_files = []
    for name, dir in zip(args.names, args.indirs):
        with open(os.path.join(dir, 'predictions.pkl'), 'rb') as f:
            predictions, test_files = pickle.load(f)
            all_predictions[name] = predictions
            all_test_files.append(tuple(test_files))

    test_files_set = set(all_test_files)
    if len(test_files_set) != 1:
        raise RuntimeError('Test files are different.')

    [paths] = test_files_set 
    df = read_data(paths, all_predictions)

    for subdir in ['distributions', 'flavours', 'response', 'resolution', 'residual']:
        try:
            os.makedirs(os.path.join(args.outdir, subdir, 'png'))
            os.makedirs(os.path.join(args.outdir, subdir, 'pdf'))
        except FileExistsError:
            pass
    
    names = ['Standard'] + args.names
    plot_distrs(df, names, os.path.join(args.outdir, 'distributions'))
    compare_flavours(df, names, os.path.join(args.outdir, 'flavours'))

    binning = np.geomspace(20, 3000, 20)
    bin_centers = np.sqrt(binning[:-1] * binning[1:])

    for (ieta, eta_bin), (flavour_label, flavour_ids) in itertools.product(
        enumerate([(0, 2.5), (2.5, 5)], start=1),
        [
            ('uds', {1, 2, 3}), ('c', {4}), ('b', {5}), ('g', {21}),
            ('all', {0, 1, 2, 3, 4, 5, 21})
        ]
    ):
        df_bin = df[
            (np.abs(df.eta_gen) >= eta_bin[0])
            & (np.abs(df.eta_gen) < eta_bin[1])
            & df.flavour.isin(flavour_ids)
        ]
        bins = df_bin.groupby(pd.cut(df_bin.pt_gen, binning))

        plot_median_response(
            os.path.join(args.outdir, 'response'),
            flavour_label, bins, bin_centers, eta_bin, ieta, names
        )

        plot_resolution(
            os.path.join(args.outdir, 'resolution'),
            flavour_label, bins, bin_centers, eta_bin, ieta, names
        )
    
    for (ieta, eta_bin), flavours in itertools.product(
        enumerate([(0, 2.5), (2.5, 5)], start=1),
        itertools.combinations([('uds', {1, 2, 3}), ('c', {4}), ('b', {5}), ('g', {21})], r=2),
    ):
        bins = []
        for i, flavour_ids in enumerate([flavours[0][1], flavours[1][1]]):
            df_bin = df[
                (np.abs(df.eta_gen) >= eta_bin[0])
                & (np.abs(df.eta_gen) < eta_bin[1])
                & df.flavour.isin(flavour_ids)
            ]
            bins.append(df_bin.groupby(pd.cut(df_bin.pt_gen, binning)))

        plot_median_residual(
            os.path.join(args.outdir, 'residual'),
            bin_centers, (flavours[0][0], flavours[1][0]), bins, eta_bin, ieta, names
        )
