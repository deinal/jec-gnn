import os
import glob
import warnings
import argparse
import itertools
import awkward as ak
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
PFNanoAODSchema.warn_missing_crossrefs = False


JET_FEATURES = ['pt', 'eta', 'phi', 'mass', 'area', 'qgl_axis2', 'qgl_ptD', 'qgl_mult', 'partonFlavour']
PF_FEATURES = ['pt', 'eta', 'phi', 'd0', 'dz', 'd0Err', 'dzErr', 'trkChi2', 'vtxChi2', 'puppiWeight', 'puppiWeightNoLep', 'charge', 'lostInnerHits', 'pdgId', 'pvAssocQuality', 'trkQuality']

CATEGORICAL_FEATURES = {
    'jet': {
        'partonFlavour': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 21]
    },
    'pf': {
        'charge': [-1, 0, 1],
        'lostInnerHits': [-1, 0, 1, 2],
        'pdgId': [-211, -13, -11, 1, 2, 11, 13, 22, 130, 211],
        'pvAssocQuality': [0, 1, 4, 5, 6, 7],
        'trkQuality': [0, 1, 5]
    }
}


def read_nanoaod(path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='found duplicate branch')
        events = NanoEventsFactory.from_root(path, schemaclass=PFNanoAODSchema).events()

    jets = events.Jet[(ak.count(events.Jet.matched_gen.pt, axis=1) >= 2)]

    sorted_jets = jets[ak.argsort(jets.matched_gen.pt, ascending=False, axis=1)]

    leading_jets = ak.concatenate((sorted_jets[:,0], sorted_jets[:,1]), axis=0)

    selected_jets = leading_jets[(leading_jets.matched_gen.pt > 15) & (abs(leading_jets.matched_gen.eta) < 5)]

    valid_jets = selected_jets[~ak.is_none(selected_jets.matched_gen.pt)]

    for field in ['dz', 'dzErr', 'd0', 'd0Err']:
        valid_jets = valid_jets[ak.all(valid_jets.constituents.pf[field] != np.inf, axis=1)]

    return valid_jets, valid_jets.constituents.pf


def one_hot_encode(array, categories):
    cardinality = len(categories)
    category_map = dict(zip(categories, range(cardinality)))
    for i, val in enumerate(array):
        array[i] = category_map[val]
    return np.eye(cardinality)[array]


def preprocess(jet, pf, jet_features, pf_features):
    # Preprocess numerical features
    jet['target'] = jet.matched_gen.pt / jet.pt
    jet_features.append('target')
    jet['log_pt'] = np.log(jet.pt)
    jet_features.append('log_pt')
    pf['rel_eta'] = (pf.eta - jet.eta) * np.sign(jet.eta)
    pf_features.append('rel_eta')
    pf['rel_pt'] = pf.pt / jet.pt
    pf_features.append('rel_pt')
    pf['rel_phi'] = (pf.phi - jet.phi + np.pi) % (2 * np.pi) - np.pi
    pf_features.append('rel_phi')
    
    # One hot encode categorical features
    for key, categories in CATEGORICAL_FEATURES['jet'].items():
        encoded_matrix = one_hot_encode(np.array(jet[key]), categories)
        for i in range(len(categories)):
            field = f'{key}_{i}'
            jet[field] = encoded_matrix[:,i]
            jet_features.append(field)
    
    counts = ak.num(pf)
    flat_pf = ak.flatten(pf)
    for key, categories in CATEGORICAL_FEATURES['pf'].items():
        encoded_matrix = one_hot_encode(np.array(flat_pf[key]), categories)
        for i in range(len(categories)):
            field = f'{key}_{i}'
            flat_pf[field] = encoded_matrix[:,i]
            pf_features.append(field)
    pf = ak.unflatten(flat_pf, counts)
    
    # Select gen level features for result plots
    jet['gen_pt'] = jet.matched_gen.pt
    jet_features.append('gen_pt')
    jet['gen_eta'] = jet.matched_gen.eta
    jet_features.append('gen_eta')
    jet['gen_partonFlavour'] = jet.matched_gen.partonFlavour
    jet_features.append('gen_partonFlavour')
    jet['gen_hadronFlavour'] = jet.matched_gen.hadronFlavour
    jet_features.append('gen_hadronFlavour')
    
    return jet[jet_features], pf[pf_features]


def create_dataset(root_file, parquet_dir):
    print(parquet_dir)
    
    jet, pf = read_nanoaod(root_file)
    jet, pf = preprocess(jet, pf, JET_FEATURES.copy(), PF_FEATURES.copy())
    
    try:
        os.makedirs(parquet_dir)
    except FileExistsError:
        pass
    
    ak.to_parquet(jet, os.path.join(parquet_dir, 'jet.parquet'))
    ak.to_parquet(pf, os.path.join(parquet_dir, 'pf.parquet'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-i', '--in_dir', required=True, help='Directory containing jet data')
    arg_parser.add_argument('-o', '--out_dir', required=True, help='Directory to store preprocessed data')
    args = arg_parser.parse_args()

    root_files = glob.glob(os.path.join(args.in_dir, '*.root'))
    num_files = len(root_files)

    try:
        os.makedirs(args.out_dir)
    except FileExistsError:
        pass

    with ProcessPoolExecutor(max_workers=10) as executor:
        parquet_dirs = ['/'.join((path, str(index))) for index, path in enumerate(itertools.repeat(args.out_dir, num_files), start=1)]
        results = executor.map(create_dataset, root_files, parquet_dirs)
