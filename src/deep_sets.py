from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Dense, TimeDistributed, BatchNormalization, Dropout, Concatenate, Add
from src.layers import Sum


def get_deep_sets(num_ch, num_ne, num_sv, num_globals, config):
    ch_constituents = Input(shape=(None, num_ch), ragged=True, name='charged constituents')

    ne_constituents = Input(shape=(None, num_ne), ragged=True, name='neutral constituents')

    sv_constituents = Input(shape=(None, num_sv), ragged=True, name='secondary vertices')

    ch_head = _deep_sets(ch_constituents, config, prefix='ch')

    ne_head = _deep_sets(ne_constituents, config, prefix='ne')

    sv_head = _deep_sets(sv_constituents, config, prefix='sv')

    globals = Input(shape=(num_globals,), name='globals')

    inputs_head = Concatenate(name='head')([ch_head, ne_head, sv_head, globals])

    if config['type'] == 'mlp':
        x = _mlp(inputs_head, config, name='head')
    if config['type'] == 'resnet':
        x = _resnet(inputs_head, config, name='head')

    outputs = Dense(1, name='head_dense_output')(x)

    model = Model(inputs=[ch_constituents, ne_constituents, sv_constituents, globals], outputs=outputs, name='dnn')

    model.summary()

    for layer in model.layers:
        if isinstance(layer, TimeDistributed):
            layer.layer.summary()

    return model


def _deep_sets(constituents, config, prefix):
    constituents_slice = Input(shape=(constituents.shape[-1],), name=f'{prefix}_constituents_slice')

    if config['type'] == 'mlp':
        deepset_outputs_slice = _mlp(constituents_slice, config, name=f'{prefix}_deepset')
    if config['type'] == 'resnet':
        deepset_outputs_slice = _resnet(constituents_slice, config, name=f'{prefix}_deepset')

    deepset_model_slice = Model(inputs=constituents_slice, outputs=deepset_outputs_slice, name=f'{prefix}_deepset_model_slice')

    deepset_outputs = TimeDistributed(deepset_model_slice, name=f'{prefix}_deepset_distributed')(constituents)

    constituents_head = Sum(axis=1, name=f'{prefix}_constituents_head')(deepset_outputs)

    return constituents_head


def _mlp(x, config, name):
    for idx, units in enumerate(config['units'], start=1):
        x = Dense(units, kernel_initializer=config['initializer'], name=f'{name}_dense_{idx}')(x)
        if config['batch_norm']:
            x = BatchNormalization(name=f'{name}_batch_normalization_{idx}')(x)
        x = Activation(config['activation'], name=f'{name}_activation_{idx}')(x)
        if config['dropout']:
            x = Dropout(config['dropout'], name=f'{name}_dropout_{idx}')(x)
    return x


def _resnet(x, config, name):
    units = config['units']
    for idx in range(0, len(units) - 1, 2):
        n1 = units[idx]
        n2 = units[idx + 1]
        layer_idx = idx // 2 + 1

        x_main = Dense(n1, kernel_initializer=config['initializer'], name=f'{name}_dense_{layer_idx}_1')(x)
        x_main = Activation(config['activation'], name=f'{name}_activation_{layer_idx}_1')(x_main)
        x_main = Dense(n2, kernel_initializer=config['initializer'], name=f'{name}_dense_{layer_idx}_2')(x_main)

        # Include a projection to match the dimensions
        sc = Dense(n2, kernel_initializer=config['initializer'], use_bias=False, name=f'{name}_projection_{layer_idx}')(x)

        x = Add(name=f'add_{layer_idx}')([x_main, sc])
        x = Activation(config['activation'], name=f'{name}_activation_{layer_idx}_2')(x)

    if len(units) % 2 == 1:
        x = Dense(units[-1], kernel_initializer=config['initializer'], name=f'{name}_dense_last')(x)
        x = Activation(config['activation'], name=f'{name}_activation_last')(x)

    return x
