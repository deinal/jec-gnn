#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import tensorflow as tf
import argparse
import pickle
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from src.deep_sets import get_deep_sets
from src.particle_net import get_particle_net
from src.data import create_datasets


def calculate_num_features(features, categories):
    num_ch = sum([
        len(features['ch']['numerical']),
        sum([len(categories[field]) for field in features['ch']['categorical']]),
        len(features['ch']['synthetic'])
    ])
    num_ne = sum([
        len(features['ne']['numerical']),
        sum([len(categories[field]) for field in features['ne']['categorical']]),
        len(features['ne']['synthetic'])
    ])
    num_sv = sum([
        len(features['sv']['numerical']),
        sum([len(categories[field]) for field in features['sv']['categorical']]),
        len(features['sv']['synthetic'])
    ])
    num_globals = sum([
        len(features['jet']['numerical']),
        sum([len(categories[field]) for field in features['jet']['categorical']]),
        len(features['jet']['synthetic'])
    ])

    return num_ch, num_ne, num_sv, num_globals


def get_loss():
    def loss_fn(y_true, y_pred):
        # Avoid spikes in loss while training by not taking unreasonable response values into account
        mask = tf.math.logical_and(y_true > -1, y_true < 1)
        return tf.math.reduce_mean(tf.math.abs(y_pred - y_true) * tf.cast(mask, tf.float32))
    return loss_fn


def get_callbacks(config):
    # Reduce learning rate when nearing convergence
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=config['reduce_lr_on_plateau']['factor'], 
        patience=config['reduce_lr_on_plateau']['patience'], min_lr=config['reduce_lr_on_plateau']['min_lr'],
        mode='auto', min_delta=config['reduce_lr_on_plateau']['min_delta'], cooldown=0, verbose=1
    )
    # Stop early if the network stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=config['early_stopping']['min_delta'], 
        patience=config['early_stopping']['patience'], mode='auto', baseline=None, 
        restore_best_weights=True, verbose=1
    )

    return [reduce_lr_on_plateau, early_stopping]


def plot_loss(history, outdir):
    """Plot training loss."""

    x = range(1, len(history['loss'][1:]) + 1)
    plt.plot(x, history['loss'][1:], label='Training loss')
    plt.plot(x, history['val_loss'][1:], label='Validation loss')

    changes = np.where(np.roll(history['lr'], 1) != history['lr'])[0]
    ymin, ymax = plt.gca().get_ylim()
    plt.vlines(changes[1:] - 1, ymin=ymin, ymax=ymax, ls='dashed', lw=0.8, colors='gray')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'loss.png'))
    plt.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-i', '--indir', required=True, help='Directory containing jet data')
    arg_parser.add_argument('-o', '--outdir', required=True, help='Where to store outputs')
    arg_parser.add_argument('-c', '--config', required=True, help='Config file')
    arg_parser.add_argument('--gpus', nargs='+', required=True, help='GPUs to run on in the form 0 1 etc.')
    arg_parser.add_argument('--save-model', action='store_true', help='If model should be saved')
    arg_parser.add_argument('--save-weights', action='store_true', help='If weights should be saved')
    args = arg_parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpus)
    print('GPU devices:', tf.config.list_physical_devices('GPU'))

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    with open(args.config) as f:
        config = yaml.safe_load(f)
        net = config['net']

    shutil.copyfile(args.config, f'{args.outdir}/config.yaml')

    train_ds, val_ds, test_ds, metadata = create_datasets(net, args.indir, config['data'])

    num_ch, num_ne, num_sv, num_globals = calculate_num_features(config['data']['features'], config['data']['transforms']['categorical'])

    train_ds = train_ds.shuffle(config['shuffle_buffer'])

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if net == 'deep_sets':
            dnn = get_deep_sets(
                num_ch, num_ne, num_sv, num_globals,
                config['model']['deep_sets']
            )
        if net == 'particle_net':
            dnn = get_particle_net(
                num_ch, num_ne, num_sv, num_globals, metadata['num_points'], 
                config['model']['particle_net']
            )

        dnn.compile(optimizer=config['optimizer'], loss=get_loss())
        dnn.optimizer.lr.assign(config['lr'])

    tf.keras.utils.plot_model(dnn, os.path.join(args.outdir, 'model.png'), dpi=100, show_shapes=True, expand_nested=True)

    callbacks = get_callbacks(config['callbacks'])

    fit = dnn.fit(train_ds, validation_data=val_ds, epochs=config['epochs'], callbacks=callbacks)

    test_loss = dnn.evaluate(test_ds)
    print('Test loss:', test_loss)

    start = time.time()
    predictions = dnn.predict(test_ds)
    end = time.time()
    inference_time = (end - start)
    print('Inference time:', inference_time)

    plot_loss(fit.history, args.outdir)
    
    # Save predictions and corresponding test files
    with open(os.path.join(args.outdir, 'predictions.pkl'), 'wb') as f:
        pickle.dump((predictions, metadata['test_files']), f)

    # Save training history
    with open(os.path.join(args.outdir, 'history.pkl'), 'wb') as f:
        pickle.dump(fit.history, f)

    # Save model
    if args.save_model:
        dnn.save(os.path.join(args.outdir, 'dnn'))

    # Save weights
    if args.save_weights:
        dnn.save_weights(os.path.join(args.outdir, 'weights'))
