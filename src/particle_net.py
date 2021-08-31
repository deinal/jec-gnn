import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, Layer, Multiply, Concatenate
from src.layers import Mean, Max, Expand, Squeeze


def get_particle_net(num_ch, num_ne, num_sv, num_globals, num_points, config):
    """
    ParticleNet: Jet Tagging via Particle Clouds
    arxiv.org/abs/1902.08570
    
    Parameters
    ----------
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """

    ch_fts = Input(name='charged constituents', shape=(num_points['ch'], num_ch))
    ch_mask = Input(name='charged mask', shape=(num_points['ch'], 1))
    ch_coord_shift = Input(name='charged coord shift', shape=(num_points['ch'], 1))
    ch_points = Input(name='charged points', shape=(num_points['ch'], 2))

    ne_fts = Input(name='neutral constituents', shape=(num_points['ne'], num_ne))
    ne_mask = Input(name='neutral mask', shape=(num_points['ne'], 1))
    ne_coord_shift = Input(name='neutral coord shift', shape=(num_points['ne'], 1))
    ne_points = Input(name='neutral points', shape=(num_points['ne'], 2))

    sv_fts = Input(name='secondary vertices', shape=(num_points['sv'], num_sv))
    sv_mask = Input(name='sv mask', shape=(num_points['sv'], 1))
    sv_coord_shift = Input(name='sv coord shift', shape=(num_points['sv'], 1))
    sv_points = Input(name='sv points', shape=(num_points['sv'], 2))
    
    globals = Input(name='globals', shape=(num_globals,))

    outputs = _particle_net_base(
        ch_fts, ch_mask, ch_coord_shift, ch_points,
        ne_fts, ne_mask, ne_coord_shift, ne_points,
        sv_fts, sv_mask, sv_coord_shift, sv_points,
        globals, config, num_points
    )

    model = Model(
        inputs=[
            ch_fts, ch_mask, ch_coord_shift, ch_points,
            ne_fts, ne_mask, ne_coord_shift, ne_points,
            sv_fts, sv_mask, sv_coord_shift, sv_points,
            globals
        ], outputs=outputs
    )

    model.summary()

    return model


def _particle_net_base(
        ch_fts, ch_mask, ch_coord_shift, ch_points,
        ne_fts, ne_mask, ne_coord_shift, ne_points,
        sv_fts, sv_mask, sv_coord_shift, sv_points,
        globals, config, num_points
    ):
    """
    points : (N, P, C_coord)
    features:  (N, P, C_features), optional
    mask: (N, P, 1), optional
    """

    ch_pool = _constituent_block(ch_fts, ch_coord_shift, ch_points, ch_mask, num_points, config, prefix='ch')
    ne_pool = _constituent_block(ne_fts, ne_coord_shift, ne_points, ne_mask, num_points, config, prefix='ne')
    sv_pool = _constituent_block(sv_fts, sv_coord_shift, sv_points, sv_mask, num_points, config, prefix='sv')

    x = Concatenate(name='head')([ch_pool, ne_pool, sv_pool, globals])

    for layer_idx, units in enumerate(config['units'], start=1):
        x = Dense(units, name=f'dense_{layer_idx}')(x)
        x = Activation(config['activation'], name=f'activation_{layer_idx}')(x)
        if config['dropout']:
            x = Dropout(config['dropout'], name=f'dropout_{layer_idx}')(x)
    out = Dense(1, name='out')(x)
    return out # (N, num_classes)


def _constituent_block(fts, coord_shift, points, mask, num_points, config, prefix):
    for layer_idx, channels in enumerate(config[prefix]['channels'], start=1):
        pts = Add(name=f'{prefix}_add_{layer_idx}')([coord_shift, points]) if layer_idx == 1 else Add(name=f'{prefix}_add_{layer_idx}')([coord_shift, fts])
        fts = _edge_conv(
            pts, fts, num_points[prefix], channels, config[prefix]['K'], config, name=f'{prefix}_edge_conv_{layer_idx}'
        )
    fts = Multiply()([fts, mask])
    pool = Mean(axis=1)(fts) # (N, C)
    return pool


def _edge_conv(points, features, num_points, channels, K, config, name):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    fts = features
    knn_fts = KNearestNeighbors(num_points, K, name=f'{name}_knn')([points, fts])

    x = knn_fts
    for idx, channel in enumerate(channels, start=1):
        x = Conv2D(
            channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
            use_bias=False if config['batch_norm'] else True, kernel_initializer=config['initializer'], name=f'{name}_conv_{idx}'
        )(x)
        if config['batch_norm']:
            x = BatchNormalization(name=f'{name}_batchnorm_{idx}')(x)
        if config['activation']:
            x = Activation(config['activation'], name=f'{name}_activation_{idx}')(x)

    if config['pooling'] == 'max':
        fts = Max(axis=2, name=f'{name}_max')(x) # (N, P, C')
    else:
        fts = Mean(axis=2, name=f'{name}_mean')(x) # (N, P, C')

    if config['shortcut']:
        sc = Expand(axis=2, name=f'{name}_shortcut_expand')(features)
        sc = Conv2D(
            channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
            use_bias=False if config['batch_norm'] else True, kernel_initializer=config['initializer'], name=f'{name}_shortcut_conv'
        )(sc)
        if config['batch_norm']:
            sc = BatchNormalization(name=f'{name}_shortcut_batchnorm')(sc)
        sc = Squeeze(axis=2, name=f'{name}_shortcut_squeeze')(sc)

        x = Add(name=f'{name}_add')([sc, fts])
    else:
        x = fts

    return Activation(config['activation'], name=f'{name}_activation')(x) # (N, P, C')


class KNearestNeighbors(Layer):
    def __init__(self, num_points, k, **kwargs):
        super().__init__(**kwargs)
        self.num_points = num_points
        self.k = k

    def call(self, inputs):
        points, features = inputs
        # distance
        D = _batch_distance_matrix_general(points, points) # (N, P, P)
        _, top_k_indices = tf.math.top_k(-D, k=self.k + 1) # (N, P, K+1)
        top_k_indices = top_k_indices[:, :, 1:] # (N, P, K)

        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, self.num_points, self.k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(top_k_indices, axis=3)], axis=3) # (N, P, K, 2)
        
        knn_fts =  tf.gather_nd(features, indices) # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, self.k, 1)) # (N, P, K, C)

        return tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1) # (N, P, K, 2*C)


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def _batch_distance_matrix_general(A, B):
    r_A = tf.math.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.math.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.linalg.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D
