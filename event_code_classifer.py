"""
Black box assessment of signals in LCLS data using random forest classifiers to predict event code labels from smalldata azimuthal integration results
"""

from joblib import Parallel,delayed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from time import time
from typing import Tuple,List
start = time()

from argparse import ArgumentParser
parser = ArgumentParser(__doc__)
parser.add_argument('--folds', type=int, default=10, help="Number of crossvalidation folds with default 10")
parser.add_argument('--num-jobs', type=int, default=-1, help="Number of cpu cores to use. By default use all available.")
parser.add_argument('--image', type=str, default=None, help="Save a plot of the results to this file.")
parser.add_argument('--csv', type=str, default=None, help="Save a csv of the results to this file.")
parser.add_argument('--show', action='store_true', help="Show the results plot interactively.")
parser.add_argument('--event-codes', nargs='+', type=int, help="For h5 inputs, specify the event codes of interest.")
parser.add_argument('data_file', help="A .npz or .h5 data file.")
parser = parser.parse_args()

def load_npz_file(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    label_names, labels = np.unique(data['label'], return_inverse=True)
    x = np.concat((
        data['q'],
        data['i'],
    ), axis=-1)
    data = (
        x,
        labels,
    )
    return data,label_names

def load_h5_file(h5_file, event_codes):
    """
    Load a smalldata, psana2 file
    """
    import h5py as h5
    f = h5.File(h5_file)
    x = np.concat((
        np.squeeze(f['jungfrau']['azav_azav']), #has an extra dimension
        f['jungfrau']['azav_q'], #doesn't have any extra dimensions
        ), axis=-1
    )
    labels = []
    label_names = []
    for c in eventcodes:
        label_names.append(f"Event Code {c}")
        labels.append(
            f['timing']['eventcodes'][:][:, c] == 1
        )
    labels = np.column_stack(labels)
    data = (
        x,
        labels
    )
    return data, label_names

def load_data_file(data_file : str, **h5_kwargs) -> Tuple[Tuple['x' : np.ndarray, 'labels' : np.ndarray], 'label_names' : list[str]]:
    """
    Extract training data for the random forest classifier from an input file. 

    Parameters
    ----------
    data_file : str
        The input .npz file from A. Wolff or .h5 file from smalldata

    Returns
    -------
    (x, labels) : (array, array)
        Returns an array x (num_events, num_dimensions) to be used to learn a function to predict the integers in labels, (num_events,)
    label_names : list
        Human-readable list of label names corresponding to the integers in labels.
    """
    if data_file.endswith('.npz'):
        return load_npz_file(data_file)
    elif data_file.endswith('.h5'):
        return load_h5_file(data_file, **h5_kwargs)
    else:
        raise ValueError(f"data_file, {data_file}, must be either a numpy `.npz` or an hdf5 `.h5` file")

def get_auc_record(fold_id, data, fold_ids):
    """
    Helper method to train crossvalidation folds in parallel
    """
    idx = ( fold_ids == fold_id )
    test  = [d[idx] for d in data]
    train = [d[~idx] for d in data]

    n = RandomForestClassifier()
    n.fit(*train)
    labels_true = test[1]
    scores = n.predict_proba(test[0])
    record = {
        'Multi-Class (One vs One)' : metrics.roc_auc_score(labels_true, scores, multi_class='ovo'),
        'Multi-Class (One vs Rest)' : metrics.roc_auc_score(labels_true, scores, multi_class='ovr'),
    }
    for l in range(scores.shape[-1]):
        record[f'{label_names[l]}'] = metrics.roc_auc_score(labels_true == l, scores[...,l])
    return record

folds = parser.folds
num_jobs = parser.num_jobs

seed = 42
data,label_names = load_data_file(parser.data_file)

l = data[0].shape[0]
fold_ids = np.random.choice(folds, l)

roc_data = Parallel(num_jobs)(delayed(get_auc_record)(fold_id, data, fold_ids) for fold_id in range(folds))
roc_data = pd.DataFrame.from_records(roc_data)

report =  pd.DataFrame(
    roc_data.mean().apply(lambda x: "{:0.3f}".format(x)) + \
    roc_data.std().apply(lambda x: "±{:0.3f}".format(x))
).rename(columns={0: "ROC AUC (mean±std)"})

sns.barplot(roc_data, color='grey')
tick_pos,tick_labels = plt.xticks()
plt.xticks(
    tick_pos,
    tick_labels,
    ha='right',
    rotation_mode='anchor',
    rotation=45,
)
plt.ylabel("Area Under Receiver Operating Characteristic")
plt.grid(which='both', axis='y', ls='--', color='k')
plt.title(parser.data_file)
plt.tight_layout()
print(report)
stop = time()
print(f"Runtime: {stop - start} seconds")

if parser.csv is not None:
    report.to_csv(parser.csv)
if parser.image is not None:
    plt.savefig(parser.image)
if parser.show:
    plt.show()

