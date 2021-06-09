import glob
import warnings
import tensorflow as tf
import awkward as ak
import numpy as np
import math
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema


def create_datasets(net, indir, config):
    root_paths = glob.glob(f'{indir}/*.root')
    num_files = len(root_paths)
    train_split = int(config['train_size'] * num_files)
    test_split = int(config['test_size'] * num_files) + train_split
    
    train_files = root_paths[:train_split]
    test_files = root_paths[train_split:test_split]
    val_files = root_paths[test_split:]

    train = _create_dataset(
        net, train_files, config['features'], config['batch_size'], 
        config['num_points'], config['transforms']
    )
    test = _create_dataset(
        net, test_files, config['features'], config['batch_size'], 
        config['num_points'], config['transforms']
    )
    val = _create_dataset(
        net, val_files, config['features'], config['batch_size'], 
        config['num_points'], config['transforms']
    )
    
    metadata = _get_metadata(
        config['features'], config['transforms']['categorical'], config['num_points']
    )

    return train, val, test, test_files, metadata


def _get_metadata(features, category_map, num_points):
    num_constituents = sum([
        len(features['pf_cands']['numerical']),
        sum([
            len(category_map[f'pf_cand_{field}']) for field in features['pf_cands']['categorical']
        ])
    ])
    num_globals = sum([
        len(features['jets']['numerical']),
        sum([
            len(category_map[f'jet_{field}']) for field in features['jets']['categorical']
        ])
    ])

    return (num_constituents, num_globals, num_points)


def _create_dataset(net, files, features, batch_size, num_points, transforms):
    dataset = tf.data.Dataset.from_tensor_slices(files)

    dataset = dataset.map(
        lambda path: _retrieve_data(
            net, path, num_points, features['jets']['numerical'], features['jets']['categorical'], 
            features['pf_cands']['numerical'], features['pf_cands']['categorical']
        ),
        num_parallel_calls=4 # a fixed number instead of autotune limits the RAM usage
    )

    tables = _create_category_tables(transforms['categorical'])

    dataset = dataset.map(
        lambda data, target: (_preprocess(
                net, data, tables, transforms, 
                features['jets']['numerical'], features['jets']['categorical'], 
                features['pf_cands']['numerical'], features['pf_cands']['categorical'], 
                features['pf_cands']['synthetic']
            ),
            target
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.unbatch().batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _preprocess(
        net, data, tables, transforms, global_numerical, global_categorical, 
        constituent_numerical, constituent_categorical, constituent_synthetic
    ):

    # Create synthetic features
    if 'rel_pt' in constituent_synthetic:
        data['pf_cand_rel_pt'] = data['pf_cand_pt'] / tf.expand_dims(data['jet_pt'], axis=1)
        constituent_numerical = list(filter(lambda field: field != 'pt', constituent_numerical))
        data.pop('pf_cand_pt')

    if 'rel_eta' in constituent_synthetic:
        jet_eta = tf.expand_dims(data['jet_eta'], axis=1)
        data['pf_cand_rel_eta'] = (data['pf_cand_eta'] - jet_eta) * tf.math.sign(jet_eta)
        constituent_numerical = list(filter(lambda field: field != 'eta', constituent_numerical))
        data.pop('pf_cand_eta')

    if 'rel_phi' in constituent_synthetic:
        phi_diff = data['pf_cand_phi'] - tf.expand_dims(data['jet_phi'], axis=1)
        data['pf_cand_rel_phi'] = (phi_diff + math.pi) % (2 * math.pi) - math.pi
        constituent_numerical = list(filter(lambda field: field != 'phi', constituent_numerical))
        data.pop('pf_cand_phi')

    # Create ParticleNet inputs
    if net == 'particlenet':
        mask = tf.cast(tf.math.not_equal(data['pf_cand_rel_eta'], 0), dtype=tf.float32) # 1 if valid
        coord_shift = tf.multiply(1e6, tf.cast(tf.math.equal(mask, 0), dtype=tf.float32))
        points = tf.concat([data['pf_cand_rel_eta'], data['pf_cand_rel_phi']], axis=2)

    # Transform the data
    for name, transform in transforms['numerical'].items():
        if name in data:
            if transform == 'log':
                data[name] = tf.math.log(data[name])
            elif transform == 'abs':
                data[name] = tf.math.abs(data[name])
            else:
                raise RuntimeError(f'Unknown transform {transform} for {name}.')

    for name, categories in transforms['categorical'].items():
        if name in data:
            encoded_feature = _one_hot_encode(data[name], tables[name], categories)
            if name.startswith('jet'):
                data[name] = tf.squeeze(encoded_feature, axis=1) # Remove excess dimension created by tf.one_hot
            if name.startswith('pf_cand'):
                data[name] = tf.squeeze(encoded_feature, axis=2)

    # Concatenate the data
    globals = tf.concat(
        [data[f'jet_{field}'] for field in global_numerical + global_categorical], axis=1
    )
    constituents = tf.concat(
        [data[f'pf_cand_{field}'] for field in 
        constituent_numerical + constituent_categorical + constituent_synthetic], axis=2
    )
    
    # Mind the order of the inputs when constructing the model!
    if net == 'deepset':
        inputs = (constituents, globals)
    if net == 'particlenet':
        inputs = (points, constituents, mask, coord_shift, globals)

    return inputs


def _one_hot_encode(feature, table, categories):
    cardinality = len(categories)

    # Map integer categories to an ordered list e.g. charge from [-1, 0, 1] to [0, 1, 2]
    if isinstance(feature, tf.RaggedTensor):
        feature = tf.ragged.map_flat_values(lambda x: table.lookup(x), feature)
    else:
        feature = table.lookup(feature)

    # One-hot encode categories to orthogonal vectors e.g. [0, 1, 2] to [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return tf.one_hot(tf.cast(feature, tf.int32), depth=cardinality, dtype=tf.float32)


def _create_category_tables(category_map):
    tables = {}

    for name, categories in category_map.items():
        keys_tensor = tf.constant(categories)
        vals_tensor = tf.range(len(categories))

        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
            default_value=-1
        )

        tables[name] = table

    return tables


def _retrieve_data(
        net, path, num_points, global_numerical, global_categorical, 
        constituent_numerical, constituent_categorical
    ):

    global_names = [
        f'jet_{field}' for field in global_numerical + global_categorical
    ]
    constituent_names = [
        f'pf_cand_{field}' for field in constituent_numerical + constituent_categorical
    ]
    names = ['target'] + global_names + constituent_names

    inp = [
        net, path, num_points, global_numerical, global_categorical, 
        constituent_numerical, constituent_categorical
    ]
    Tout = (
        [tf.float32] +
        [tf.float32] * len(global_numerical) +
        [tf.int32] * len(global_categorical) +
        [tf.float32] * len(constituent_numerical) +
        [tf.int32] * len(constituent_categorical)
    )

    if net == 'deepset':
        Tout.append(tf.int32)
        names.append('row_lengths')

    data = tf.numpy_function(_read_nanoaod, inp=inp, Tout=Tout)

    data = {key: value for key, value in zip(names, data)}

    target = data.pop('target')
    target.set_shape((None,))

    for name in global_names:
        # Shape from <unknown> to (None,)
        data[name].set_shape((None,))
        # Shape from (None,) to (None, 1)
        data[name] = tf.expand_dims(data[name], axis=1)

    if net == 'deepset':
        row_lengths = data.pop('row_lengths')
        row_lengths.set_shape((None,))

        for name in constituent_names:
            # Shape from <unknown> to (None,)
            data[name].set_shape((None,))
            # shape from (None,) to (None, None)
            data[name] = tf.RaggedTensor.from_row_lengths(data[name], row_lengths=row_lengths)
            # Shape from (None, None) to (None, None, 1)
            data[name] = tf.expand_dims(data[name], axis=2)
    
    if net == 'particlenet':
        for name in constituent_names:
            # Shape from <unknown> to (None, P)
            data[name].set_shape((None, num_points))
            # Shape from (None, P) to (None, P, 1)
            data[name] = tf.expand_dims(data[name], axis=2)

    return (data, target)


def _read_nanoaod(
        net, path, num_points, global_numerical, global_categorical,
        constituent_numerical, constituent_categorical
    ):

    # Decode bytestrings
    net = net.decode()
    path = path.decode()
    global_numerical = [field.decode() for field in global_numerical]
    global_categorical = [field.decode() for field in global_categorical]
    constituent_numerical = [field.decode() for field in constituent_numerical]
    constituent_categorical = [field.decode() for field in constituent_categorical]

    valid_jets = retrieve_jets(path)

    target = valid_jets.matched_gen.pt / valid_jets.pt

    globals = valid_jets[global_numerical + global_categorical]

    constituents = valid_jets.constituents.pf[constituent_numerical + constituent_categorical]
    
    data = [ak.to_numpy(target).astype(np.float32)]

    for field in global_numerical:
        data.append(ak.to_numpy(globals[field]).astype(np.float32))

    for field in global_categorical:
        data.append(ak.to_numpy(globals[field]).astype(np.int32))

    if net == 'deepset':
        row_lengths = ak.num(constituents, axis=1)
        flat_constituents = ak.flatten(constituents, axis=1)

        for field in constituent_numerical:
            data.append(ak.to_numpy(flat_constituents[field]).astype(np.float32))

        for field in constituent_categorical:
            data.append(ak.to_numpy(flat_constituents[field]).astype(np.int32))

        data.append(ak.to_numpy(row_lengths).astype(np.int32))

    if net == 'particlenet':
        for field in constituent_numerical:
            none_padded_constituent = ak.pad_none(constituents[field], target=num_points, clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.float32))

        for field in constituent_categorical:
            none_padded_constituent = ak.pad_none(constituents[field], target=num_points, clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.int32))

    return data


def retrieve_jets(path):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='found duplicate branch')
        warnings.filterwarnings('ignore', message='missing cross-reference index')
        events = NanoEventsFactory.from_root(path, schemaclass=PFNanoAODSchema).events()

    jets = events.Jet[(ak.count(events.Jet.matched_gen.pt, axis=1) >= 2)]

    sorted_jets = jets[ak.argsort(jets.matched_gen.pt, ascending=False, axis=1)]

    leading_jets = ak.concatenate((sorted_jets[:,0], sorted_jets[:,1]), axis=0)

    selected_jets = leading_jets[(leading_jets.matched_gen.pt > 30) & (abs(leading_jets.matched_gen.eta) < 2.5)]

    valid_jets = selected_jets[~ak.is_none(selected_jets.matched_gen.pt)]

    return valid_jets