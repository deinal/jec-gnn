import tensorflow as tf
from train import calculate_num_features
from src.data import create_dataset
from src.deep_sets import get_deep_sets
from src.particle_net import get_particle_net
import argparse
import pickle
import yaml
import glob
import os

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-m', '--model_dir', required=True, help='Model directory')
    arg_parser.add_argument('-d', '--data_dir', required=True, help='Data directory')
    arg_parser.add_argument('-p', '--pred_dir', required=True, help='Prediction directory')
    args = arg_parser.parse_args()

    with open(os.path.join(args.model_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
        net = config['net']
        cd = config['data']

    files = glob.glob(os.path.join(args.data_dir, '*.root'))
    
    num_ch, num_ne, num_sv, num_globals = calculate_num_features(cd['features'], cd['transforms']['categorical'])

    if net == 'deep_sets':
        dnn = get_deep_sets(
            num_ch, num_ne, num_sv, num_globals,
            config['model']['deep_sets']
        )
    if net == 'particle_net':
        num_points = {
                'ch': cd['features']['ch']['num_points'],
                'ne': cd['features']['ne']['num_points'],
                'sv': cd['features']['sv']['num_points']
        }
        dnn = get_particle_net(
            num_ch, num_ne, num_sv, num_globals, num_points, 
            config['model']['particle_net']
        )

    dnn.load_weights(os.path.join(args.model_dir, 'weights'))
    print(dnn.summary())

    ds = create_dataset(net, files, cd['features'], cd['batch_size'], cd['transforms'])

    predictions = dnn.predict(ds)

    with open(os.path.join(args.pred_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
