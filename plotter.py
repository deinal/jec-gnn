import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(paths, predictions):
    dfs = []
    for path in paths:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)
        df = data['globals'][['Jet_pt', 'GenJet_pt', 'GenJet_partonFlavour', 'GenJet_hadronFlavour', 'GenJet_eta']]
        flavour = df.GenJet_hadronFlavour.where(df.GenJet_hadronFlavour != 0, other=np.abs(df.GenJet_partonFlavour))
        df = df.drop(columns=['GenJet_partonFlavour', 'GenJet_hadronFlavour'])
        df['flavour'] = flavour
       
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    df['response'] = df.Jet_pt / df.GenJet_pt
    df['dnn_response'] = predictions

    return df


def plot_loss(outdir, history):
    plt.plot(history['loss'][1:], label='Training loss')
    plt.plot(history['val_loss'][1:], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./results/{outdir}/loss.pdf')


def get_binned_statistics(variable, binning_variable, binning):
    indices = np.digitize(binning_variable, binning)
    bin_mean = np.empty_like(binning)
    bin_std = np.empty_like(binning)
    for i in range(len(binning)):
        bin_data = variable[indices == i+1]
        bin_mean[i] = bin_data.mean()
        bin_std[i] = bin_data.std()
    return bin_mean, bin_std


def plot_mean_response(outdir, df, flavour):
    
    binning = np.linspace(200, 2200, 20)
    bin_centers = binning + (binning[1] - binning[0]) / 2.0

    response_mean, response_std = get_binned_statistics(df.response, df.GenJet_pt, binning)
    dnn_response_mean, dnn_response_std = get_binned_statistics(df.dnn_response, df.GenJet_pt, binning)

    fig, ax = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(10, 6),
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1}
    )
    fig.suptitle('Mean ' + flavour + '-jet response w.r.t. gen p$_{T}$')

    ax[0].errorbar(bin_centers, dnn_response_mean, yerr=dnn_response_std, fmt='o', label='DNN')
    ax[0].errorbar(bin_centers, response_mean, yerr=response_std, fmt='o', label='Standard')
    ax[0].axhline(1, ls='dashed', c='gray')
    ax[0].set_ylabel("Mean response")
    ax[0].legend()

    ax[1].hist(df.GenJet_pt, bins=binning)
    ax[1].set_ylabel("Jets/bin")
    ax[1].set_xlabel("gen p$_{T}$")

    fig.savefig(f'./results/{outdir}/{flavour}_mean_response.pdf')


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    outdir = config['outdir']

    with open(f'./results/{outdir}/history.pkl', 'rb') as f:
        history = pickle.load(f)

    plot_loss(outdir, history)

    with open(f'./results/{outdir}/predictions.pkl', 'rb') as f:
        predictions, test_files = pickle.load(f)

    df = read_data(test_files, predictions)

    for flavour, ids in [
        ('uds', {1, 2, 3}), ('b', {5}), ('g', {21}), ('all', {0, 1, 2, 3, 4, 5, 21})
    ]:
        plot_mean_response(outdir, df[df.flavour.isin(ids)], flavour)