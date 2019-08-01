import warnings
warnings.filterwarnings("ignore")

import argparse
from datetime import datetime
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [25, 25]
matplotlib.rcParams['legend.fontsize'] =  13
matplotlib.rcParams['legend.handlelength'] = 1
matplotlib.rcParams['legend.markerscale'] = 1

from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor

classifiers_dict = {'classification': {'logistic_regression': LogisticRegression(multi_class='auto', solver='newton-cg'),
                                       'gaussian': GaussianNB(),
                                       'svm': SVC(gamma='auto'),
                                       'xgb': XGBClassifier()},
                    'regression': {'linear': LinearRegression(),
                                   'lasso': Lasso(),
                                   'xgb': XGBRegressor()}}

import pandas as pd
import numpy as np
import os

def clf_report(ground_truth, preds):
    acc = accuracy_score(ground_truth, preds)
    precision = precision_score(ground_truth, preds, average='macro')
    f1 = f1_score(ground_truth, preds, average='macro')

    return {'accuracy': acc,
            'precision': precision,
            'f1': f1}

def generate_features(df):
    return np.array([np.array(xi) for xi in pd.to_datetime(df).apply(lambda x: [x.year, x.month, x.day, x.hour, x.minute, x.second, x.weekday(),])])

def run(fpath):
    now = datetime.now()
    timestamp = '{0}/{1}/{2}/{0}{1}{2}{3}{4}_'.format(now.year, now.month, now.day, now.hour, now.minute)

    fname = fpath.split('/')[-1].split('.')[0].lower()

    print(f"Loading file {fpath}")
    df = pd.read_csv(fpath)

    count_df = df.groupby(by='source_name').count().reset_index()[['source_name', 'timestamp']]

    df_ignore = count_df[count_df['timestamp'] < 50]

    os.makedirs('logs/data/', exist_ok=True)
    with open(f'logs/data/{fname}_ignored_sources.txt', 'w') as fp:
        for source in df_ignore['source_name'].values:
            fp.write(source)
            fp.write('\n')

    df = df[~df['source_name'].isin(df_ignore['source_name'])][['id', 'source_name', 'value', 'timestamp']]
    sources = df['source_name'].unique()
    
    d = {'number of sources': len(sources)}
    encoder_dict = {}
    np.random.seed(8821)

    le = LabelEncoder()

    for source in sources[:2]:
        ids = df[df['source_name'] == source]['id'].unique()
        for idx, id_ in enumerate(ids[:2], start=1):
            if idx % 25 == 0 or idx == len(ids):
                print(f"{idx} / {len(ids)}")
            aux_df = df[(df['source_name'] == source) & (df['id'] == id_)]
            X, y = generate_features(aux_df['timestamp']), aux_df['value']
            if len(y.unique()) < 2:
                continue
            try:
                y = y.astype(int)
            except:
                le.fit(y)
                y = pd.Series(le.transform(y))
                encoder_dict[source] = {k: v for k, v in zip(y, aux_df['value'])}
                with open(os.path.join('logs/train', 'encoder.json'), 'w') as fp:
                    json.dump(encoder_dict, fp)

            total_examples = len(y)
            unique_examples_ratio = len(y.unique()) / total_examples
            if unique_examples_ratio > 0.9:
                model_type = 'regression'
            else:
                model_type = 'classification'

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

            d[source] = {'train size': len(X_train), 'test size': len(X_test), 'model type': model_type}

            classifiers = classifiers_dict[model_type]
            res = {}
            for name, clf in classifiers.items():
                try:
                    clf.fit(X_train, y_train)
                except:
                    continue
                preds = clf.predict(X_test)
                report = clf_report(y_test, preds)
                res[name] = report
            best = sorted([(k, v['f1']) for k, v in res.items()], key=lambda x: -x[1])[0]
            model = classifiers[best[0]]
            os.makedirs('models', exist_ok=True)
            os.makedirs('plots', exist_ok=True)
            with open(os.path.join('models', fname + '_' + source.replace(' ', '_') + '_' + id_ + '.model'), 'wb') as fp:
                pickle.dump(model, fp)
            with open(os.path.join('logs/train', fname + '_' + source.replace(' ', '_') + '_' + id_ + '_models.logs'), 'w') as fp:
                json.dump(res, fp)
            print(source)
            print(id_)
            print(len(y_test))
            print()

            h = pd.to_datetime(pd.DataFrame(X_test[:, :-1], columns=['year', 'month', 'day', 'hour', 'minute', 'second']))
            plt.plot(h, y_test, 'r|', h, preds, 'g_', markersize=11)
            plt.gcf().autofmt_xdate()
            plt.gca().legend(('ground truth','prediction'))
            plt.savefig(os.path.join('plots', fname + '_' + source.replace(' ', '_') + '_' + id_ + '.png'))
            plt.clf()

    os.makedirs('logs/train', exist_ok=True)
    with open(os.path.join('logs/train', 'statistics.json'), 'w') as fp:
        json.dump(d, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='data/dataset.csv')

    args = parser.parse_args()

    fname = args.f
    run(fname)
