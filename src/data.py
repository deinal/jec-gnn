import os
import glob
import tensorflow as tf
import awkward as ak
import numpy as np


def create_datasets(net, indir, config):
    parquet_dirs = glob.glob(os.path.join(indir, '*'))
    num_dirs = len(parquet_dirs)
    train_split = int(config['train_size'] * num_dirs)
    test_split = int(config['test_size'] * num_dirs) + train_split
    
    train_dirs = parquet_dirs[:train_split]
    test_dirs = parquet_dirs[train_split:test_split]
    val_dirs = parquet_dirs[test_split:]

    jet_categorical = []
    for key, categories in config['features']['jet']['categorical'].items():
        jet_categorical.extend([f'{key}_{i}' for i in range(len(categories))])
    config['features']['jet']['categorical'] = jet_categorical

    pf_categorical = []
    for key, categories in config['features']['pf']['categorical'].items():
        pf_categorical.extend([f'{key}_{i}' for i in range(len(categories))])
    config['features']['pf']['categorical'] = pf_categorical

    train_ds = _create_dataset(
        net, train_dirs, config['features'],
        config['num_points'], config['batch_size']
    )
    test_ds = _create_dataset(
        net, test_dirs, config['features'],
        config['num_points'], config['batch_size']
    )
    val_ds = _create_dataset(
        net, val_dirs, config['features'],
        config['num_points'], config['batch_size']
    )
    
    metadata = {
        'train_dirs': train_dirs,
        'test_dirs': test_dirs,
        'val_dirs': val_dirs,
        'num_points': config['num_points']
    }

    return train_ds, val_ds, test_ds, metadata


def _create_dataset(net, paths, features, num_points, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(
        lambda path: _retrieve_data(
            net, path, num_points, features['jet'], features['pf']
        ),
        num_parallel_calls=tf.data.AUTOTUNE # a fixed number instead of autotune limits the RAM usage
    )
    dataset = dataset.map(
        lambda data, target: (
            _prepare_inputs(
                net, data, features['jet'], features['pf']
            ),
            target
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.unbatch().batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _prepare_inputs(net, data, jet, pf):
    globals = tf.concat([data[f'jet_{field}'] for field in jet['numerical'] + jet['categorical']], axis=1)
    constituents = tf.concat([data[f'pf_{field}'] for field in pf['numerical'] + pf['categorical']], axis=2)

    # Create ParticleNet inputs
    if net == 'particlenet':
        data['mask'] = tf.cast(tf.math.not_equal(data['pf_rel_eta'], 0), dtype=tf.float32) # 1 if valid
        data['coord_shift'] = tf.multiply(1e6, tf.cast(tf.math.equal(data['mask'], 0), dtype=tf.float32))
        data['points'] = tf.concat([data['pf_rel_eta'], data['pf_rel_phi']], axis=2)
        inputs = (constituents, globals, data['points'], data['coord_shift'], data['mask'])
    if net == 'deepset':
        inputs = (constituents, globals)

    return inputs


def _retrieve_data(net, path, num_points, jet, pf):
    global_names = [
        f'jet_{field}' for field in jet['numerical'] + jet['categorical']
    ]
    constituent_names = [
        f'pf_{field}' for field in pf['numerical'] + pf['categorical']
    ]
    names = ['target'] + global_names + constituent_names

    inp = [
        net, path, num_points, jet['numerical'], jet['categorical'], 
        pf['numerical'], pf['categorical']
    ]
    Tout = (
        [tf.float32] + 
        [tf.float32] * len(jet['numerical']) +
        [tf.float32] * len(jet['categorical']) +
        [tf.float32] * len(pf['numerical']) +
        [tf.float32] * len(pf['categorical'])
    )

    if net == 'deepset':
        Tout.append(tf.int32)
        names.append('row_lengths')

    data = tf.numpy_function(_retrieve_np_data, inp=inp, Tout=Tout)

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

    return (data, target) #, sample_weights)


def _retrieve_np_data(
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

    globals, constituents = read_parquet(path)

    data = [ak.to_numpy(globals.target).astype(np.float32)]

    for field in global_numerical:
        data.append(ak.to_numpy(globals[field]).astype(np.float32))

    for field in global_categorical:
        data.append(ak.to_numpy(globals[field]).astype(np.float32))

    if net == 'deepset':
        row_lengths = ak.num(constituents, axis=1)
        flat_constituents = ak.flatten(constituents, axis=1)

        for field in constituent_numerical:
            data.append(ak.to_numpy(flat_constituents[field]).astype(np.float32))

        for field in constituent_categorical:
            data.append(ak.to_numpy(flat_constituents[field]).astype(np.float32))

        data.append(ak.to_numpy(row_lengths).astype(np.int32))

    if net == 'particlenet':
        for field in constituent_numerical:
            none_padded_constituent = ak.pad_none(constituents[field], target=num_points, clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.float32))

        for field in constituent_categorical:
            none_padded_constituent = ak.pad_none(constituents[field], target=num_points, clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.float32))

    return data


def read_parquet(path):
    jet = ak.from_parquet(os.path.join(path, 'jet.parquet'))
    pf = ak.from_parquet(os.path.join(path, 'pf.parquet'))

    return jet, pf
