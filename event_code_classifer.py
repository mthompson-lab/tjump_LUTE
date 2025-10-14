"""
Black box assessment of signals in LCLS data using random forest classifiers to predict event code labels from smalldata azimuthal integration results
"""

from joblib import Parallel,delayed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import os
from time import time
from typing import Tuple,List
start = time()

from argparse import ArgumentParser
parser = ArgumentParser(__doc__)
parser.add_argument('--folds', type=int, default=10, help="Number of crossvalidation folds with default 10")
parser.add_argument('--num-jobs', type=int, default=-1, help="Number of cpu cores to use. By default use all available.")
parser.add_argument('--output-dir', type=str, default=".", help="Save the results to this directory. Default is the current directory.")
parser.add_argument('--image', type=str, default=None, help="Save a plot of the results to this file (relative to output_dir). Default is None.")
parser.add_argument('--csv', type=str, default=None, help="Save a csv of the results to this file (relative to output_dir). Default is None.")
parser.add_argument('--show', action='store_true', help="Show the results plot interactively.")
parser.add_argument('--event-codes', nargs='+', type=int, help="For h5 inputs, specify the event codes of interest.")
parser.add_argument('--data-file', help="A .npz or .h5 data file.")
parser = parser.parse_args()

# Set matplotlib backend based on whether we're showing plots interactively
if not parser.show:
    matplotlib.use('Agg')

def load_npz_file(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    label_names, labels = np.unique(data['label'], return_inverse=True)
    x = np.concatenate((
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
    Load a smalldata, depending on the psana version
    """
    import h5py as h5
    f = h5.File(h5_file)
    xray_on = f['lightStatus']['xray'][:] == 1

    # Determine psana version for h5 keys
    if 'jungfrau' in f.keys():
        psana_version = 2
        azint_method = 'azav'
        detname = 'jungfrau'
        print(f"Detector jungfrau found in h5 key. Assuming psana version 2, and azimuthal integration method {azint_method}.")
    else:
        psana_version = 1
        azint_method = 'pyfai'
        detname = 'Rayonix'
        print(f"Detector jungfrau not found in h5 key. Assuming psana version 1, with detector {detname} and azimuthal integration method {azint_method}.")

    # Load azimuthal integration data
    azav = f[detname][f'{azint_method}_azav'][xray_on]
    if len(azav.shape) == 3:
        azav = np.average(azav, axis=1) #2d azimuthal average
    elif len(azav.shape) == 2:
        pass
    else:
        raise ValueError(f"azav shape error: {azav.shape}")
    q = f[detname][f'{azint_method}_q'][xray_on]
    x = np.concatenate((azav, q), axis=-1) # shape (num_events, num_q * 2)
    
    # Load event code labels
    labels = []
    label_names = [f"Event Code {c}" for c in event_codes]
    print(f"label_names: {label_names}")
    if psana_version == 1:
        for c in event_codes:
            labels.append(
                f['evr'][f'code_{c}'][:][xray_on] == 1
            )
    else:
        for c in event_codes:
            labels.append(
                f['timing']['eventcodes'][:][xray_on][:, c] == 1
            )
    labels = np.column_stack(labels) # one-hot encoded, shape (num_events, num_event_codes)
    labels = np.argmax(labels, axis=1) # shape (num_events,)
    data = (
        x,
        labels
    )
    return data, label_names

def load_data_file(data_file : str, **h5_kwargs) -> Tuple[Tuple[np.ndarray, np.ndarray], list[str]]:
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
    record = {}
    for l in range(scores.shape[-1]):
        record[f'{label_names[l]}'] = metrics.roc_auc_score(labels_true == l, scores[...,l])
    if scores.shape[-1] == 2: scores = scores[:, 1] # for binary classification, we only need the positive class
    record['Multi-Class (One vs One)'] = metrics.roc_auc_score(labels_true, scores, multi_class='ovo')
    record['Multi-Class (One vs Rest)'] = metrics.roc_auc_score(labels_true, scores, multi_class='ovr')
    return record

folds = parser.folds
num_jobs = parser.num_jobs

seed = 42
data,label_names = load_data_file(parser.data_file, event_codes=parser.event_codes)

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
plt.title(parser.data_file, fontsize=8)
plt.tight_layout()
print(report)
stop = time()
print(f"Runtime: {stop - start} seconds")

if parser.csv is not None:
    report.to_csv(os.path.join(parser.output_dir, parser.csv))
if parser.image is not None:
    plt.savefig(os.path.join(parser.output_dir, parser.image))
if parser.show:
    plt.show()

