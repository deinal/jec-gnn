import os
import glob
import tensorflow as tf
import awkward as ak
import numpy as np
import uproot

def create_datasets(net, indir, config):
    root_files = glob.glob(os.path.join(indir, '*.root'))
    num_files = len(root_files)
    train_split = int(config['train_size'] * num_files)
    test_split = int(config['test_size'] * num_files) + train_split
    
    train_files = root_files[:train_split]
    test_files = root_files[train_split:test_split]
    val_files = root_files[test_split:]

    train_ds = create_dataset(
        net, train_files, config['features'],
        config['batch_size'], config['transforms']
    )
    test_ds = create_dataset(
        net, test_files, config['features'],
        config['batch_size'], config['transforms']
    )
    val_ds = create_dataset(
        net, val_files, config['features'],
        config['batch_size'], config['transforms']
    )
    
    metadata = {
        'train_files': train_files,
        'test_files': test_files,
        'val_files': val_files,
        'num_points': {
            'ch': config['features']['ch']['num_points'],
            'ne': config['features']['ne']['num_points'],
            'sv': config['features']['sv']['num_points']
        }
    }

    return train_ds, val_ds, test_ds, metadata


def create_dataset(net, paths, features, batch_size, transforms):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(
        lambda path: _retrieve_data(
            net, path, features['jet'], features['ch'], features['ne'], features['sv']
        ),
        num_parallel_calls=tf.data.AUTOTUNE # a fixed number instead of autotune limits the RAM usage
    )

    tables = _create_category_tables(transforms['categorical'])
    
    dataset = dataset.map(
        lambda data, target: (
            _prepare_inputs(
                net, data, features['jet'], features['ch'], features['ne'], features['sv'], transforms, tables
            ),
            target
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.unbatch().batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def _prepare_inputs(net, data, jet, ch, ne, sv, transforms, tables):
    pt = tf.expand_dims(data['pt'], axis=1)
    eta = tf.expand_dims(data['eta'], axis=1)
    phi = tf.expand_dims(data['phi'], axis=1)

    pf_pt = tf.concat((data['ch_pt'], data['ne_pt']), axis=1)
    pf_eta = tf.concat((data['ch_eta'], data['ne_eta']), axis=1)
    pf_phi = tf.concat((data['ch_phi'], data['ne_phi']), axis=1)

    if 'ch_rel_pt' in ch['synthetic']:
        data['ch_rel_pt'] = data['ch_pt'] / pt
    if 'ne_rel_pt' in ne['synthetic']:
        data['ne_rel_pt'] = data['ne_pt'] / pt
    if 'sv_rel_pt' in sv['synthetic']:
        data['sv_rel_pt'] = data['sv_pt'] / pt
    
    if 'ch_rel_eta' in ch['synthetic']:
        data['ch_rel_eta'] = (data['ch_eta'] - eta) * tf.math.sign(eta)
    if 'ne_rel_eta' in ne['synthetic']:
        data['ne_rel_eta'] = (data['ne_eta'] - eta) * tf.math.sign(eta)
    if 'sv_rel_eta' in sv['synthetic']:
        data['sv_rel_eta'] = (data['sv_eta'] - eta) * tf.math.sign(eta)

    if 'ch_rel_phi' in ch['synthetic']:
        data['ch_rel_phi'] = (data['ch_phi'] - phi + np.pi) % (2 * np.pi) - np.pi
    if 'ne_rel_phi' in ne['synthetic']:
        data['ne_rel_phi'] = (data['ne_phi'] - phi + np.pi) % (2 * np.pi) - np.pi
    if 'sv_rel_phi' in sv['synthetic']:
        data['sv_rel_phi'] = (data['sv_phi'] - phi + np.pi) % (2 * np.pi) - np.pi

    if 'log_pt' in jet['synthetic']:
        data['log_pt'] = tf.math.log(data['pt'])

    # quark / gluon discrimination features
    if 'mult' in jet['synthetic']:
        data['mult'] = tf.math.reduce_sum(tf.cast(pf_pt > 1., tf.float32), axis=1)
    if 'ptD' in jet['synthetic']:
        data['ptD'] = tf.math.sqrt(tf.reduce_sum(pf_pt**2, axis=1)) / tf.reduce_sum(pf_pt, axis=1)
    if 'axis2' in jet['synthetic']:
        deta = pf_eta - eta
        dphi = pf_phi - phi
        weight = pf_pt**2

        sum_weight = tf.reduce_sum(weight, axis=1)
        ave_deta = tf.reduce_sum(deta * weight, axis=1) / sum_weight
        ave_dphi = tf.reduce_sum(dphi * weight, axis=1) / sum_weight
        ave_deta2 = tf.reduce_sum(deta**2 * weight, axis=1) / sum_weight
        ave_dphi2 = tf.reduce_sum(dphi**2 * weight, axis=1) / sum_weight
        sum_detadphi = tf.reduce_sum(deta * dphi * weight, axis=1)

        a = ave_deta2 - ave_deta * ave_deta
        b = ave_dphi2 - ave_dphi * ave_dphi
        c = -(sum_detadphi / sum_weight - ave_deta * ave_dphi)

        delta = tf.math.sqrt(tf.math.abs((a - b) * (a - b) + 4 * c**2))

        data['axis2'] = tf.where(a + b - delta > 0, tf.math.sqrt(0.5 * (a + b - delta)), 0)

    for field, categories in transforms['categorical'].items():
        if field in data:
            encoded_feature = _one_hot_encode(data[field], tables[field], categories)
            data[field] = tf.squeeze(encoded_feature, axis=2)

    globals = tf.concat([data[field] for field in jet['numerical'] + jet['categorical'] + jet['synthetic']], axis=1)
    ch_constituents = tf.concat([data[field] for field in ch['numerical'] + ch['categorical'] + ch['synthetic']], axis=2)
    ne_constituents = tf.concat([data[field] for field in ne['numerical'] + ne['categorical'] + ne['synthetic']], axis=2)
    secondary_vertices = tf.concat([data[field] for field in sv['numerical'] + sv['categorical'] + sv['synthetic']], axis=2)

    # Create ParticleNet inputs
    if net == 'particle_net':
        ch_mask, ch_coord_shift, ch_points = _construct_particle_net_inputs(data['ch_rel_eta'], data['ch_rel_phi'])
        ne_mask, ne_coord_shift, ne_points = _construct_particle_net_inputs(data['ne_rel_eta'], data['ne_rel_phi'])
        sv_mask, sv_coord_shift, sv_points = _construct_particle_net_inputs(data['sv_rel_eta'], data['sv_rel_phi'])
        inputs = (
            ch_constituents, ch_mask, ch_coord_shift, ch_points,
            ne_constituents, ne_mask, ne_coord_shift, ne_points,
            secondary_vertices, sv_mask, sv_coord_shift, sv_points,
            globals
        )

    if net == 'deep_sets':
        inputs = (ch_constituents, ne_constituents, secondary_vertices, globals)

    return inputs


def _construct_particle_net_inputs(rel_eta, rel_phi):
    mask = tf.cast(tf.math.not_equal(rel_eta, 0), dtype=tf.float32) # 1 if valid
    coord_shift = tf.multiply(1e6, tf.cast(tf.math.equal(mask, 0), dtype=tf.float32))
    points = tf.concat([rel_eta, rel_phi], axis=2)
    return mask, coord_shift, points


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


def _retrieve_data(net, path, jet, ch, ne, sv):
    jet_fields = jet['numerical'] + jet['categorical']
    ch_fields = ch['numerical'] + ch['categorical']
    ne_fields = ne['numerical'] + ne['categorical']
    sv_fields = sv['numerical'] + sv['categorical']

    pf = {
        'numerical': ch['numerical'] + ne['numerical'] + sv['numerical'],
        'categorical': ch['categorical'] + ne['categorical'] + sv['categorical']
    }
    pf_fields = pf['numerical'] + pf['categorical']

    fields = ['target'] + jet_fields + pf_fields
    inp = [
        net, path, jet['numerical'], jet['categorical'], 
        pf['numerical'], pf['categorical'],
        ch['num_points'], ne['num_points'], sv['num_points']
    ]
    Tout = (
        [tf.float32] + 
        [tf.float32] * len(jet['numerical']) +
        [tf.int32] * len(jet['categorical']) +
        [tf.float32] * len(pf['numerical']) +
        [tf.int32] * len(pf['categorical'])
    )

    if net == 'deep_sets':
        Tout.extend([tf.int32, tf.int32, tf.int32])
        fields.extend(['ch_size', 'ne_size', 'sv_size'])

    data = tf.numpy_function(_retrieve_np_data, inp=inp, Tout=Tout)

    data = {key: value for key, value in zip(fields, data)}

    target = data.pop('target')
    target.set_shape((None,))

    for field in jet_fields:
        # Shape from <unknown> to (None,)
        data[field].set_shape((None,))
        # Shape from (None,) to (None, 1)
        data[field] = tf.expand_dims(data[field], axis=1)

    if net == 'deep_sets':
        ch_size = data.pop('ch_size')
        ch_size.set_shape((None,))

        for field in ch_fields:
            # Shape from <unknown> to (None,)
            data[field].set_shape((None,))
            # shape from (None,) to (None, None)
            data[field] = tf.RaggedTensor.from_row_lengths(data[field], row_lengths=ch_size)
            # Shape from (None, None) to (None, None, 1)
            data[field] = tf.expand_dims(data[field], axis=2)
        
        ne_size = data.pop('ne_size')
        ne_size.set_shape((None,))

        for field in ne_fields:
            # Shape from <unknown> to (None,)
            data[field].set_shape((None,))
            # shape from (None,) to (None, None)
            data[field] = tf.RaggedTensor.from_row_lengths(data[field], row_lengths=ne_size)
            # Shape from (None, None) to (None, None, 1)
            data[field] = tf.expand_dims(data[field], axis=2)
        
        sv_size = data.pop('sv_size')
        sv_size.set_shape((None,))

        for field in sv_fields:
            # Shape from <unknown> to (None,)
            data[field].set_shape((None,))
            # shape from (None,) to (None, None)
            data[field] = tf.RaggedTensor.from_row_lengths(data[field], row_lengths=sv_size)
            # Shape from (None, None) to (None, None, 1)
            data[field] = tf.expand_dims(data[field], axis=2)
    
    if net == 'particle_net':
        for field in ch_fields:
            # Shape from <unknown> to (None, P)
            data[field].set_shape((None, ch['num_points']))
            # Shape from (None, P) to (None, P, 1)
            data[field] = tf.expand_dims(data[field], axis=2)

        for field in ne_fields:
            # Shape from <unknown> to (None, P)
            data[field].set_shape((None, ne['num_points']))
            # Shape from (None, P) to (None, P, 1)
            data[field] = tf.expand_dims(data[field], axis=2)
        
        for field in sv_fields:
            # Shape from <unknown> to (None, P)
            data[field].set_shape((None, sv['num_points']))
            # Shape from (None, P) to (None, P, 1)
            data[field] = tf.expand_dims(data[field], axis=2)

    return (data, target)


def _retrieve_np_data(
        net, path, global_numerical, global_categorical,
        constituent_numerical, constituent_categorical,
        ch_num_points, ne_num_points, sv_num_points
    ):
    # Decode bytestrings
    net = net.decode()
    path = path.decode()
    global_numerical = [field.decode() for field in global_numerical]
    global_categorical = [field.decode() for field in global_categorical]
    constituent_numerical = [field.decode() for field in constituent_numerical]
    constituent_categorical = [field.decode() for field in constituent_categorical]
    num_points = {'ch': ch_num_points, 'ne': ne_num_points, 'sv': sv_num_points}

    jets = uproot.open(path)['Jets'].arrays()

    target = np.log(jets.pt_gen / jets.pt)

    data = [ak.to_numpy(target).astype(np.float32)]

    for field in global_numerical:
        data.append(ak.to_numpy(jets[field]).astype(np.float32))

    for field in global_categorical:
        data.append(ak.to_numpy(jets[field]).astype(np.float32))

    if net == 'deep_sets':
        for field in constituent_numerical:
            data.append(ak.to_numpy(ak.flatten(jets[field])).astype(np.float32))

        for field in constituent_categorical:
            data.append(ak.to_numpy(ak.flatten(jets[field])).astype(np.int32))

        data.append(ak.to_numpy(jets.ch_size).astype(np.int32))
        data.append(ak.to_numpy(jets.ne_size).astype(np.int32))
        data.append(ak.to_numpy(jets.sv_size).astype(np.int32))

    if net == 'particle_net':
        for field in constituent_numerical:
            prefix = field[:2]
            none_padded_constituent = ak.pad_none(jets[field], target=num_points[prefix], clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.float32))

        for field in constituent_categorical:
            prefix = field[:2]
            none_padded_constituent = ak.pad_none(jets[field], target=num_points[prefix], clip=True, axis=1)
            zero_padded_constituent = ak.to_numpy(none_padded_constituent).filled(0)
            data.append(zero_padded_constituent.astype(np.int32))

    return data
