import torch
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import os
import pandas as pd
from typing import Dict, Any
import yaml
import pickle
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter


def plot_prediction_hist(classification_predictions, use_logspace=True, use_percentage=False):
    predicted_classes = np.unique(classification_predictions).size
    plt.figure(figsize=(10, 8), dpi=150)
    plt.hist(classification_predictions, 
             edgecolor='black', 
             bins=np.linspace(0, 69, 70),
             weights=np.ones_like(classification_predictions) / len(classification_predictions))
    if use_logspace:
        plt.gca().set_yscale("log")
        ticks = [1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
        ticks_labels = [Text(0, t, f'{t}') for t in ticks]
        plt.yticks(ticks, ticks_labels)
    plt.xlabel('Predicted classes [0-69]')
    if use_percentage:
        plt.ylabel('Log relative frequencies of predicted classes [%]')
        vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels([f'{i*100:.2f}%' for i in vals])
    else:
        plt.ylabel('Log relative frequencies of predicted classes')
    plt.title(f'Scaled histogram of SVM predictions - total of {predicted_classes} predicted In-Distribution classes')
    
    plt.show()


def plot_distributions(scores, detector_name, id_thresh=None, fn_detection_rate=None, save_plot_dir=None, bw=0.1, nbins=200, use_logspace=False,
                       normalize_density=False, save_plot=False):
    id_scores = scores['id'][detector_name]
    test_scores = scores['ood'][detector_name]

    min_val = min(min(id_scores), min(test_scores))
    max_val = max(max(id_scores), max(test_scores))
    xs = np.linspace(min_val, max_val, nbins)
    density_id = gaussian_kde(id_scores, bw)
    density_test = gaussian_kde(test_scores, bw)
    density_id_estimate = density_id(xs)
    density_test_estimate = density_test(xs)

    plt.figure(figsize=(6, 4), dpi=300)
    if normalize_density:
        # Make it a pdf so it integrates to 1
        bin_width = np.diff(xs)[0]
        num_observations_id = density_id_estimate.sum()
        num_observations_test = density_test_estimate.sum()
        density_id_estimate = density_id_estimate / (num_observations_id * bin_width)
        density_test_estimate = density_test_estimate / (num_observations_test * bin_width)
        assert (density_id_estimate * bin_width).sum().round() == 1, "PDF does not integrate to 1!"
        assert (density_test_estimate * bin_width).sum().round() == 1, "PDF does not integrate to 1!"

    # Quantitative experiments
    if id_thresh is None:
        plt.plot(xs, density_id_estimate, "--", color='blue', label='ID set')
        plt.plot(xs, density_test_estimate, "--", color='orange', label='OOD set')
        idx = np.argwhere(np.diff(np.signbit(density_test_estimate - density_id_estimate))).flatten()
        plt.plot(xs[idx], density_id_estimate[idx], 'rx')
        for intr in idx:
            plt.annotate(f'{xs[intr]:.3f}', (xs[intr] + 2 * (xs[1] - xs[0]), density_id_estimate[intr]))
    else:   # Qualitative experiments
        plt.plot(xs, density_id_estimate, "--", color='blue', label='Compounds dataset')
        plt.plot(xs, density_test_estimate, "--", color='orange', label='Test dataset')
        plt.axvline(id_thresh, color='red', linestyle ="-", label=f'Threshold {id_thresh:.2f}')

    plt.legend(loc='upper right')
    # plt.title(
    #     f"SVM: {detector_name} scores for OOD dataset{' (Normalized)' if normalize_density else ''}{' (y on logspace)' if use_logspace else ''}")
    plt.xlabel(f'{detector_name} scores')
    # plt.ylabel(f'{"Normalized " if normalize_density else ""}PDF estimated using Gaussian')
    plt.grid()
    if use_logspace:
        plt.gca().set_yscale("log")
        # plt.yticks([])
    if save_plot:
        suffix = f"{fn_detection_rate}" if {fn_detection_rate} is not None else ""
        plt.savefig(f"{save_plot_dir}/distribution_plot_{detector_name}_{suffix}.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def subset_dataset_from_indices(dataset, indices):
    ds = deepcopy(dataset)
    ds['data_points'] = ds['data_points'][indices]
    ds['data_labels'] = ds['data_labels'][indices]
    ds['metadata'] = ds['metadata'].iloc[indices]
    return ds


def subsample(dataset, alpha=0.01, shuffle=True):
    X = dataset['data_points']
    y = dataset['data_labels']
    _, X_test, _, y_test = train_test_split(X, y, train_size=1-alpha, stratify=y, shuffle=shuffle)
    dataset['data_points'] = X_test
    dataset['data_labels'] = y_test
    return dataset


def probs_to_logits(probs: np.array) -> torch.Tensor:
    eps = 1e-15  # small constant to avoid numerical instability
    probs = torch.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    return logits


def compute_ood_performances(labels, scores):
    # labels: 0 = OOD, 1 = ID
    # scores: it is anomality score (the higher the score, the more anomalous)

    # auroc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)

    # tnr at tpr 95
    idx_tpr95 = np.abs(tpr - .95).argmin()
    fpr_at_tpr95 = fpr[idx_tpr95]

    # dtacc (detection accuracy)
    dtacc = .5 * (tpr + 1. - fpr).max()

    # auprc (in and out)
    # ref. https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/utils.py#L91
    auprc_in = metrics.average_precision_score(y_true=labels, y_score=scores)
    auprc_out = metrics.average_precision_score(y_true=labels, y_score=-scores, pos_label=0)
    # equivalent to average_precision_score(y_true=1-labels, y_score=-scores)

    return auroc, fpr_at_tpr95, dtacc, auprc_in, auprc_out

def get_performance(scores_set, labels, classification_perf):
    pfs = {}
    pfs['auroc'] = {}
    pfs['fpr'] = {}
    pfs['dtacc'] = {}
    pfs['auin'] = {}
    pfs['auout'] = {}

    # Compute evaluation metrics
    for k in scores_set.keys():
        pfs['auroc'][k], pfs['fpr'][k], pfs['dtacc'][k], pfs['auin'][k], pfs['auout'][k] \
            = compute_ood_performances(labels, scores_set[k])

    # Show them as 100 * %
    for k_m in pfs.keys():
        for k_s in pfs[k_m].keys():
            pfs[k_m][k_s] *= 100

    # Add ID classification accuracy
    pfs['id_accuracy'] = {}
    pfs['id_precision'] = {}
    pfs['id_recall'] = {}
    for key, perf_dict in classification_perf.items():
        pfs['id_accuracy'][key] = perf_dict['accuracy']
        pfs['id_precision'][key] = perf_dict['precision']
        pfs['id_recall'][key] = perf_dict['recall']
    return pfs


def save_performance(performances, args, log_path):
    df = pd.DataFrame(performances)
    df.round(2)
    df.to_csv(log_path)
    print(f"[INFO] Performance metrics for {args.id_dataset_name} / {args.ood_dataset_name}:")
    print(df)


def create_ood_detector(ood_detector_name):
    if ood_detector_name == "msp":
        from ood_detectors.msp import MSPOODDetector as OODDetector
    elif ood_detector_name == "energy":
        from ood_detectors.energy import EnergyOODDetector as OODDetector
    elif ood_detector_name == "mahalanobis":
        from ood_detectors.mahalanobis import MahalanobisOODDetector as OODDetector
    elif ood_detector_name == "nnguide":
        from ood_detectors.nnguide import NNGuideOODDetector as OODDetector
    elif ood_detector_name == "knn":
        from ood_detectors.knn import KNNOODDetector as OODDetector
    elif ood_detector_name == "kl":
        from ood_detectors.kl import KLOODDetector as OODDetector
    elif ood_detector_name == "maxlogit":
        from ood_detectors.maxlogit import MaxLogitOODDetector as OODDetector
    elif ood_detector_name == "ssd":
        from ood_detectors.ssd import SSDOODDetector as OODDetector
    elif ood_detector_name == "vim":
        from ood_detectors.vim import VIMOODDetector as OODDetector
    else:
        raise NotImplementedError()
    return OODDetector()

def save_object(dict_obj, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_obj, f, protocol=4)

def load_object(dict_path):
    with open(dict_path, "rb") as input_file:
        return pickle.load(input_file)

def create_model(model_name):
    if model_name == "svm":
        from classification_models.svm import SVMModel as Model
    elif model_name == "cnn":
        from classification_models.cnn import CNNModel as Model
    else:
        raise NotImplementedError()
    return Model()


def import_yaml_config(yaml_path):
    # args: namespace instance
    with open(yaml_path, 'r') as stream:
        conf_yaml = yaml.safe_load(stream)
    return conf_yaml


def mkdir(path) -> None:
    """
    :param path: str - path to the directory you want to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def convert_categorical_labels(metadata, labels, str_labels=True):
    le = LabelEncoder()
    if str_labels:
        unique_classes = sorted(np.unique(metadata['compound']))
    else:
        unique_classes = sorted(np.unique(metadata))
    le.fit(unique_classes)
    return le.transform(labels)

def load_retention_times(
    data_dir='../data/id',
    dataset_name='spectrum_w_rt'
            ) -> Dict[str, Any]:
    
    retention_times = {}
    for folder_name in os.listdir(os.path.join(data_dir, dataset_name)):
        folder_path = os.path.join(data_dir, dataset_name, folder_name)
        if not os.path.isdir(folder_path):
            continue
        times_path = os.path.join(folder_path, "compounds_times.txt")
        system_n, subject_id, measurement_n = folder_name.split('_')[1:]
        with open(times_path, 'r') as times_file:
            for line in times_file:
                # Parse the line and extract compound name, rt1, and rt2
                compound_name, rt1, rt2 = line.strip().split('\t')
                
                # Convert strings to float if needed
                rt1, rt2 = float(rt1), float(rt2)
                retention_times[system_n, subject_id, measurement_n, compound_name] = [rt1, rt2]
    return retention_times

def load_dataset(
    data_dir='../data/id',
    dataset_name='spectrum_exported',
    use_retention_time=False,
    verbose=False
            ) -> Dict[str, Any]:
    """
    Load data from a directory containing .pth files.
    :param data_dir: string - Directory containing all datasets (either 'ID' or 'OOD')
    :param dataset_name: string - Name of the dataset to load ('spectrum_exported')
    :param verbose: bool - Print information about the dataset
    :return:
    X: np.array - Input data points
    y: np.array - Input data labels
    metadata: pd.DataFrame - Metadata for each data point (system, annotator_ID, measurement_number, compound)
    """

    print(f"[INFO] Loading dataset {dataset_name}.")
    start = time.time()
    X = []
    y = []

    filenames = glob.glob(os.path.join(data_dir + dataset_name, '*.pth'))        
    metadata = pd.DataFrame(list(map(lambda a: a[:-4].split('_')[2:], filenames)),
                            columns=['system', 'annotator_ID', 'measurement_number', 'compound'])
    for i, filename in enumerate(filenames):
        spectrum = torch.load(filename).numpy().flatten()
        label = metadata.iloc[i]['compound']
        X.append(spectrum)
        y.append(label)        

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # instead of string labels, use number of labels
    y = convert_categorical_labels(metadata, y)

    # load retention times
    if use_retention_time:
        retention_times = load_retention_times(data_dir)
        X_new = np.zeros((X.shape[0], X.shape[1]+2))
        X_new[:, :X.shape[1]] = X
        for i in range(X.shape[0]):
            system_n, subject_id, measurement_n, compound_name = metadata.iloc[i]
            rt1, rt2 = retention_times[system_n, subject_id, measurement_n, compound_name]
            X_new[i, -2:] = [rt1, rt2]
        X = X_new

    if verbose:
        print(f"[INFO] Dataset {dataset_name} loaded loaded in {(time.time() - start):.1f} seconds.")
        print(f"[INFO] Input data shapes: X[{X.shape}], y[{y.shape}]")
    out = {
        "data_points": X,
        "data_labels": y,
        "metadata": metadata
    }
    return out


def plot_spectrum(spectrum, metadata, save_plot=False, fig_folder='../figures/', figsize=(10, 5), dpi=100, top_n=5, verbose=0) -> None:
    """
    Plot the spectrum and annotate the top n peaks.
    :param spectrum: np.array - Spectrum to plot
    :param metadata: pd.DataFrame - Metadata for the spectrum
    :param top_n: int - Number of peaks to show on plot
    :param verbose: bool - Print information about the spectrum
    :return: None
    """
    # Find peaks in the spectrum
    peaks, _ = find_peaks(spectrum)
    # Sort peaks by height
    sorted_peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)

    # Select the n biggest peaks
    n = top_n
    top_n_peaks = sorted_peaks[:min(n, len(sorted_peaks))]
    if verbose:
        print(f"Top {n} peaks: {sorted(top_n_peaks)}")

    plt.figure(figsize=figsize, dpi=dpi)

    # set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    
    if np.round(spectrum.max()) == 1.0:
        spectrum_relative = spectrum * 100
        # Plot the spectrum
        plt.plot(spectrum_relative, linewidth=0.5, color='red', label='Spectrum')

        # Plot green points for peaks
        plt.scatter(top_n_peaks, spectrum_relative[top_n_peaks], color='green', marker='o', label='Peaks')

        # Annotate each peak with its intensity
        for peak in top_n_peaks:
            plt.annotate(f'{peak+1}', (peak+3, spectrum_relative[peak]), color='black') # indexing from 1
        plt.ylabel('Relative intensity [‰]')

    else: 
        # Plot the spectrum
        plt.plot((spectrum / spectrum.max()), linewidth=0.5, color='red', label='Spectrum')
        
        # Plot green points for peaks
        plt.scatter(top_n_peaks, spectrum[top_n_peaks], color='green', marker='o', label='Peaks')

        # Annotate each peak with its intensity
        for peak in top_n_peaks:
            plt.annotate(f'{peak+1}', (peak+2, spectrum[peak]), color='black')
        plt.ylabel('Absolute frequency')
    
    plt.xlabel('m/z')
    plt.legend()
    plt.title(f"Column #{metadata.iloc[0]}, Subject {metadata.iloc[1]}, "
              f"Measurement #{metadata.iloc[2]}\nFound compounds: {metadata.iloc[3]}")
    plt.grid()

    if save_plot:
        plt.savefig(f"{fig_folder}/spectrum_{metadata.iloc[3]}_{metadata.iloc[0]}_{metadata.iloc[1]}_{metadata.iloc[2]}.png", dpi=dpi, bbox_inches='tight')
    plt.show()


def filename_from_metadata(metadata, data_dir='../data/id/',  dataset_name='spectrum_exported') -> str:
    """
    Create a filename from the spectrum metadata.
    :param metadata: pd.DataFrame - Metadata for the spectrum
    :param data_dir: string - Directory containing all datasets (either 'ID' or 'OOD')
    :param dataset_name: string - Name of the dataset to load ('spectrum_exported')
    :return: string - Filename for the spectrum
    """
    return (f"{data_dir}{dataset_name}/"
            f"system_{metadata['system']}_{metadata['F_or_M']}_{metadata['number']}_{metadata['compound']}.pth")

