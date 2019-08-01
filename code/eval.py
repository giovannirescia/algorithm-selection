import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import pickle
import numpy as np
import os


def generate_features(points):
    """
    Generate features from timestamps
    """
    return np.array([(point.year, point.month, point.day, point.hour, point.minute, point.second, point.weekday()) for point in points])


def create_data_points(period, freq):
    """
    Generate timestamps for a given number
    of days at a given frequency (in minutes)
    """

    now = datetime.now()
    total_minutes = 60 * 24 * period
    total_points = total_minutes / freq
    points = []

    for _ in range(int(total_points)):
        now = now + timedelta(minutes = freq)
        points.append(now)

    return points


def get_models_list(models_path, filename, sources=None):
    """
    Get all the models for a dataset and sources
    """
    # if no sources are given, we use them all
    if sources is None:
        all_sources = True
    else:
        sources = sources.split(',')
        all_sources = False

    models = os.listdir(models_path)
    res = []
    sources_res = []
    for model in models:
        aux = model.split('_')
        dataset, source, id_ = aux[0], aux[1:-1], aux[-1].split('.')[0]
        source_name = ' '.join(source)
        if dataset == filename and (all_sources or source_name.lower() in sources):
            res.append(os.path.join(models_path, model))
            sources_res.append(source_name)
    return res, sources_res


def get_predictions(models, points):
    """
    Get predictions for each model and data point
    """
    points_df = pd.to_datetime(pd.DataFrame(points[:, :-1], columns=['year', 'month', 'day', 'hour', 'minute', 'second'])).to_frame('timestamp')

    res = pd.DataFrame(columns=['source', 'id', 'timestamp', 'predictions'])

    for model_path in models:
        aux = model_path.split('_')
        dataset, source, id_ = aux[0], ' '.join(aux[1:-1]), aux[-1].split('.')[0]
        aux = [(source, id_)] * points.shape[0]
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)

        preds = model.predict(points)
        aux_df = points_df.join(pd.DataFrame(preds, columns=['predictions']))
        aux_df.insert(0, column='source', value=source)
        aux_df.insert(1, column='id', value=id_)
        res = res.append(aux_df, ignore_index=True)

    return res


if __name__ == '__main__':
    now = datetime.now()
    timestamp_dir = '{0}/{1}/{2}/'.format(now.year, now.month, now.day)
    timestamp_file = '{0}{1}{2}{3}{4}_'.format(now.year, now.month, now.day, now.hour, now.minute)
    OUTPUT = os.path.join('output', timestamp_dir)
    LOGS = os.path.join('logs/eval', timestamp_dir)
    
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--models', type=str, required=True)
    parser.add_argument('--sources', type=str, default=None)
    parser.add_argument('--period', type=int, required=True)
    parser.add_argument('--frequency', type=int, required=True)

    args = parser.parse_args()

    fname = args.filename
    sources = args.sources
    period = args.period
    freq = args.frequency
    models = args.models

    models, sources = get_models_list(models, fname, sources)
    points = generate_features(create_data_points(period, freq))
    preds = get_predictions(models, points)
    preds.to_csv(os.path.join(OUTPUT, timestamp_file + fname + '_predictions.csv'), index=None)

    with open(os.path.join(LOGS, timestamp_file + 'statistics.json'), 'w') as fp:
        json.dump({'sources': ', '.join(set(sources)), 'number of data points': len(points), 'frequency': freq, 'period': period}, fp)
