import argparse
from datetime import datetime
import json
import pickle
import matplotlib
import os
import warnings
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

matplotlib.rcParams['figure.figsize'] = [25, 25]
matplotlib.rcParams['legend.fontsize'] = 13
matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.markerscale'] = 1


classifiers_dict = {'classification': {'logistic_regression': LogisticRegression(multi_class='auto', solver='newton-cg'),
                                       'gaussian': GaussianNB(),
                                       'svm': SVC(gamma='auto'),
                                       'xgb': XGBClassifier()},
                    'regression': {'linear': LinearRegression(),
                                   'lasso': Lasso(),
                                   'xgb': XGBRegressor()}}


def clf_report(ground_truth, preds, model_type):
    if model_type == 'classification':
        acc = accuracy_score(ground_truth, preds)
        precision = precision_score(ground_truth, preds, average='macro')
        f1 = f1_score(ground_truth, preds, average='macro')

        return {'accuracy': acc,
                'precision': precision,
                'f1': f1}
    else:
        mse = mean_squared_error(ground_truth, preds)
        return {'mse': mse}


def get_best_model(res, model_type):
    """
    Returns the best model according to a metric
    """
    if model_type == 'classification':
        best = sorted([(k, v['f1']) for k, v in res.items()], key=lambda x: -x[1])[0]
    elif model_type == 'regression':
        best = sorted([(k, v['mse']) for k, v in res.items()], key=lambda x: x[1])[0]

    return best


def generate_features(df):
    """
    Generate features from timestamps
    """
    return np.array([np.array(xi) for xi in pd.to_datetime(df).apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute, x.second, x.weekday()])])


def train(fpath):
    now = datetime.now()
    timestamp_dir = '{0}/{1}/{2}/'.format(now.year, now.month, now.day)
    timestamp_file = '{0}{1}{2}{3}{4}_'.format(now.year, now.month, now.day, now.hour, now.minute)
    fname = fpath.split('/')[-1].split('.')[0].lower()

    LOGS_TRAIN = os.path.join('logs/train', timestamp_dir)
    LOGS_DATA = os.path.join('logs/data', timestamp_dir)
    PLOTS = os.path.join('plots', timestamp_dir)
    MODELS = os.path.join('models', timestamp_dir)

    os.makedirs(LOGS_TRAIN, exist_ok=True)
    os.makedirs(LOGS_DATA, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    print(f"Loading file {fpath}")
    df = pd.read_csv(fpath)

    count_df = df.groupby(by='source_name').count().reset_index()[['source_name', 'timestamp']]
    # sources with less than 50 entries are not used
    df_ignore = count_df[count_df['timestamp'] < 50]
    # save ignored sources
    with open(os.path.join(LOGS_DATA, timestamp_file +  f'{fname}_ignored_sources.txt'), 'w') as fp:
        for source in df_ignore['source_name'].values:
            fp.write(source)
            fp.write('\n')

    # keep the relevant sources
    df = df[~df['source_name'].isin(df_ignore['source_name'])][['id', 'source_name', 'value', 'timestamp']]
    sources = df['source_name'].unique()

    general_statistics = {'number of sources': len(sources)}
    # to categorical data
    le = LabelEncoder()

    np.random.seed(8821)
    for source in sources:
        # separate each source by id also
        ids = df[df['source_name'] == source]['id'].unique()
        for idx, id_ in enumerate(ids, start=1):
            if idx % 25 == 0 or idx == len(ids):
                print(f"{idx} / {len(ids)}")
            aux_df = df[(df['source_name'] == source) & (df['id'] == id_)]
            # feature engineering
            X, y = generate_features(aux_df['timestamp']), aux_df['value']
            # if target column has only one unique value, it's ignored
            if len(y.unique()) < 2:
                continue
            try:
                y = y.astype(float)
            except:
                # string data to categorical
                le.fit(y)
                y = pd.Series(le.transform(y))
                encoder_dict = {k: v for k, v in zip(y, aux_df['value'])}
                with open(os.path.join(MODELS, fname + '_' + source + '_' + id_ + '_encoder.json'), 'w') as fp:
                    json.dump(encoder_dict, fp)

            total_examples = len(y)
            unique_examples_ratio = len(y.unique()) / total_examples
            # check the type of models to train
            if unique_examples_ratio > 0.8:
                model_type = 'regression'
            else:
                model_type = 'classification'
            # train and test data splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

            general_statistics[source] = {'train size': len(X_train),
                                          'test size': len(X_test),
                                          'model type': model_type}
            # list of classifiers to train for this source and id
            classifiers = classifiers_dict[model_type]
            res = {}
            for name, clf in classifiers.items():
                try:
                    clf.fit(X_train, y_train)
                except:
                    continue
                preds = clf.predict(X_test)
                report = clf_report(y_test, preds, model_type)
                res[name] = report
            # select the best model according to a metric: f1 for classification and mse for regression
            best_model = get_best_model(res, model_type)
            model = classifiers[best_model[0]]
            # save the best model for this (source, id) data
            with open(os.path.join(MODELS, fname + '_' + source.replace(' ', '_') + '_' + id_ + '.model'), 'wb') as fp:
                pickle.dump(model, fp)
            # logs for each model trained on this (source, id) data
            with open(os.path.join(LOGS_TRAIN, timestamp_file + fname + '_' + source.replace(' ', '_') + '_' + id_ + '_models.logs'), 'w') as fp:
                json.dump(res, fp)
            # save the plot of the ground truth values vs the model predictions
            h = pd.to_datetime(pd.DataFrame(X_test[:, :-1], columns=['year', 'month', 'day', 'hour', 'minute', 'second']))
            plt.plot(h, y_test, 'r|', h, preds, 'g_', markersize=11)
            plt.gcf().autofmt_xdate()
            plt.gca().legend(('ground truth', 'prediction'))
            plt.savefig(os.path.join(PLOTS, timestamp_file + fname + '_' + source.replace(' ', '_') + '_' + id_ + '.png'))
            plt.clf()

    # save the general statistics    
    with open(os.path.join(LOGS_TRAIN, timestamp_file + 'statistics.json'), 'w') as fp:
        json.dump(general_statistics, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='data/dataset.csv')

    args = parser.parse_args()
    fname = args.f

    train(fname)
