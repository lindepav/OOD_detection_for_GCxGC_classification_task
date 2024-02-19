import matplotlib.patches as mpatches
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from utils import plot_distributions


def detect_ood(scores, threshold):
    predictions = (scores >= threshold).astype(int)
    return predictions


def find_threshold(in_distribution_scores, target_detection_rate=0.95):
    # Sort scores in DESCENDING order
    sorted_scores = np.sort(in_distribution_scores)[::-1]

    # Calculate the index to achieve the target detection rate
    index = int(target_detection_rate * len(sorted_scores))

    # Choose the threshold based on the index
    threshold = sorted_scores[index]

    return threshold

def normalize_scores(in_distribution_scores, scores):
    return (scores - in_distribution_scores.min()) / in_distribution_scores.max()


def visualize_prediction(classified, ood_detector_name, custom_cmap, colors_lst, predicted_ood_ratio, fn_detection_rate,
                         save_path, plot_legend=False):
    classified = classified[ood_detector_name]
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(classified, cmap=custom_cmap)
    im_values = np.ravel(np.unique(classified)).astype(int)
    patches = [mpatches.Patch(color=colors_lst[val], label=f"Compound {val}") for val in im_values if
               val not in [70, 71]]
    patches.append(mpatches.Patch(color=colors_lst[70], label=f"OOD"))
    patches.append(mpatches.Patch(color=colors_lst[71], label=f"Invalid measurement"))
    if plot_legend:
        plt.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0., ncol=2)
    # plt.title \
    #     (f"{ood_detector_name}: {predicted_ood_ratio[ood_detector_name]:.3f}% "
    #      f"samples classified as OOD (total {len(im_values ) -2} ID classes detected)")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(aspect=str(0.2))

    # generate real retention times
    t1 = np.array(
        [6 * (i + 1) for i in range(0, 200, 1)] + [6 * 200 + 8 * (i + 1) for i, _ in enumerate(range(200, 311, 1))] + [
            6 * 200 + 8 * 111 + 10 * (i + 1) for i, _ in enumerate(range(311, classified.shape[1], 1))])
    t2 = np.array([0.005 * (i + 1) for i in range(classified.shape[0])])
    xs1 = [1, 100, 200, 311, 460]  # selected values to plot, based on the real retention times
    xs2 = [1, 500, 1000, 1500, 2000]
    t1_sampled = [t1[int(j - 1)] for j in xs1]
    t2_sampled = [t2[int(j - 1)] for j in xs2]
    plt.xticks(xs1, t1_sampled)
    plt.yticks(xs2, t2_sampled)

    plt.savefig(f'{save_path}/GCxGC_{ood_detector_name}_{fn_detection_rate}.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def plot_gcxgc_predictions(predictions, mask_valid_measurements, custom_cmap, colors_lst, save_path, plot_legend=False):
    h, w = mask_valid_measurements.shape
    gcxgc_classified = 71 * np.ones((h, w))
    gcxgc_classified[mask_valid_measurements] = predictions
    classified = gcxgc_classified
    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(classified, cmap=custom_cmap)
    im_values = np.ravel(np.unique(classified)).astype(int)
    patches = [mpatches.Patch(color=colors_lst[val], label=f"Compound {val}") for val in im_values if val not in [70, 71]]
    patches.append(mpatches.Patch(color=colors_lst[70], label=f"OOD"))
    patches.append(mpatches.Patch(color=colors_lst[71], label=f"Invalid measurement"))
    if plot_legend:
        plt.legend(handles=patches, bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0., ncol=2)
    # plt.title(f"SVM prediction without OOD detection (total {len(im_values)-2} ID classes detected)")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(aspect=str(0.2))

    # generate real retention times
    t1 = np.array(
        [6 * (i + 1) for i in range(0, 200, 1)] + [6 * 200 + 8 * (i + 1) for i, _ in enumerate(range(200, 311, 1))] + [
            6 * 200 + 8 * 111 + 10 * (i + 1) for i, _ in enumerate(range(311, classified.shape[1], 1))])
    t2 = np.array([0.005 * (i + 1) for i in range(classified.shape[0])])
    xs1 = [1, 100, 200, 311, 460]   # selected values to plot, based on the real retention times
    xs2 = [1, 500, 1000, 1500, 2000]
    t1_sampled = [t1[int(j - 1)] for j in xs1]
    t2_sampled = [t2[int(j - 1)] for j in xs2]
    plt.xticks(xs1, t1_sampled)
    plt.yticks(xs2, t2_sampled)

    plt.savefig(f'{save_path}/GCxGC.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def create_cmap_for_ood_detection(cmap_name='hsv'):
    cmap = matplotlib.colormaps[cmap_name]
    colors_lst = []
    for i in range(70):
        (r, g, b, _) = cmap(int(i * 3.5))
        colors_lst.append((r, g, b))

    # set OOD as white and invalid measurements as black
    colors_lst.append((0.99, 0.99, 0.99))
    colors_lst.append((0., 0., 0.))
    custom_cmap = LinearSegmentedColormap.from_list('diverge_72', colors_lst, N=72)
    return custom_cmap, colors_lst


def detect_and_reject_predictions(detector_scores, classification_predictions, ood_methods, ood_label=70, detection_rate=0.95):
    thresholds = {}

    num_ood_samples = len(classification_predictions)
    num_id_samples = len(detector_scores[ood_methods[0]]) - num_ood_samples

    for ood_detector_name in ood_methods:
        id_scores = detector_scores[ood_detector_name][:num_id_samples]
        thresholds[ood_detector_name] = find_threshold(id_scores, target_detection_rate=detection_rate)
        print(f"[INFO] {ood_detector_name} threshold: {thresholds[ood_detector_name]}")

    print("===================")
    predictions = {}
    predicted_ood_ratio = {}
    for ood_detector_name in ood_methods:
        ood_scores = detector_scores[ood_detector_name][num_id_samples:]
        current_predictions = detect_ood(ood_scores, thresholds[ood_detector_name])
        ood_mask = (current_predictions == 0)
        corrected_predictions = deepcopy(classification_predictions)
        corrected_predictions[ood_mask] = ood_label  # we have labels from 0 to 69, 70 is ood
        predictions[ood_detector_name] = corrected_predictions
        predicted_ood_ratio[ood_detector_name] = 100 * (
                    len(current_predictions[current_predictions == 0]) / len(current_predictions))
        print(f"[INFO] {ood_detector_name} predicts {predicted_ood_ratio[ood_detector_name]:.3f}% samples as OOD")
    return predictions, thresholds, predicted_ood_ratio


def classify_and_detect_gcxgc(detector_scores, classification_predictions, args):
    # Create a custom colormap for visualization of the results
    gcxgc = np.load(f"{args.data_root_path}/ood/{args.ood_dataset_name}/{args.ood_dataset_name}.npy").astype('float32')
    gcxgc_normalized = np.sum(gcxgc, axis=0)
    mask_valid_measurements = (gcxgc_normalized != 0)
    custom_cmap, colors_lst = create_cmap_for_ood_detection(cmap_name='hsv')

    # Visualize the classification results without OOD detectors
    plot_gcxgc_predictions(classification_predictions, mask_valid_measurements, custom_cmap, colors_lst, args.figures_save_dir_path)

    predictions, thresholds, ood_ratios = detect_and_reject_predictions(
                                                            detector_scores,
                                                            classification_predictions,
                                                            args.ood_detectors,
                                                            ood_label=70,
                                                            detection_rate=args.fn_detection_rate)

    # Create a dictionary with the classified GCxGC data for each OOD detector
    d, h, w = gcxgc.shape
    classified_gcxgc = {}
    for ood_detector_name in args.ood_detectors:
        gcxgc_classified = 71 * np.ones((h, w))
        gcxgc_classified[mask_valid_measurements] = predictions[ood_detector_name]
        classified_gcxgc[ood_detector_name] = gcxgc_classified

    # Visualize the classification results for OOD detectors
    for ood_detector_name in args.ood_detectors:
        visualize_prediction(classified_gcxgc, ood_detector_name,
                             custom_cmap, colors_lst, ood_ratios, args.fn_detection_rate, args.figures_save_dir_path)

    print(f"[INFO] Classification results saved to ./predictions/{args.ood_dataset_name} folder.")
    return thresholds


def save_performance_and_visualize(detector_scores, classification_predictions, classification_performance, log_path,
                                   args):
    # Visualize score distributions
    num_ood_samples = len(classification_predictions)
    num_id_samples = len(detector_scores[args.ood_detectors[0]]) - num_ood_samples

    id_thresholds = classify_and_detect_gcxgc(detector_scores, classification_predictions, args)

    for ood_detector_name in args.ood_detectors:
        id_scores = detector_scores[ood_detector_name][:num_id_samples]
        ood_scores = detector_scores[ood_detector_name][num_id_samples:]
        scores = {'id': {ood_detector_name: id_scores}, 'ood': {ood_detector_name: ood_scores}}
        plot_distributions(scores, ood_detector_name, id_thresh=id_thresholds[ood_detector_name], fn_detection_rate=args.fn_detection_rate, save_plot_dir=args.figures_save_dir_path, nbins=300,
                           use_logspace=args.detector_parameters['plot_distribution_logscale'],
                           normalize_density=args.detector_parameters['plot_distribution_norm'], save_plot=True)

    df = pd.DataFrame(classification_performance, index=[0])
    df.round(2)
    df.to_csv(log_path)
    print(f"[INFO] Performance metrics for {args.id_dataset_name} / {args.ood_dataset_name}:")
    print(df)