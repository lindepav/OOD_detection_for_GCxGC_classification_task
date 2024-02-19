import argparse 
import torch
from utils import mkdir, import_yaml_config, create_model, save_object, load_object, create_ood_detector, save_performance, get_performance, plot_distributions
import random
import numpy as np
from run_ood_detection import save_performance_and_visualize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str,
                        default='cnn',
                        choices=['svm', 'cnn'],
                        help='The name of model [`svm`, `cnn`]')
    parser.add_argument('--seed', type=int,
                        default=1,
                        help='Set seed number for results reproduction')
    parser.add_argument('--id_dataset_name', '-id', type=str,
                        default='spectrum_exported',
                        help='The dataset name for the in-distribution')
    parser.add_argument('--ood_dataset_name', '-ood', type=str,
                        default='uniform',
                        help='The dataset name for the out-of-distribution')
    parser.add_argument("--ood_detectors", type=str, nargs='+',
                        default=['msp', 'kl', 'maxlogit', 'energy', 'mahalanobis', 'knn', 'ssd', 'vim', 'nnguide'],
                        help="List of OOD ood_detectors")
    parser.add_argument('--num_classes', '-classes', type=int,
                        default=70,
                        help='Number of classes for the ID dataset')
    parser.add_argument('--k_fold_number', '-kfold', type=int,
                        default=5,
                        help='Parameter k for k-fold cross-validation used for training')
    parser.add_argument('--data_root_path', type=str,
                        default='./data',
                        help='Data root path')
    parser.add_argument('--save_root_path', type=str,
                        default='./saved_model_outputs',
                        help='Data path where to save model output')
    parser.add_argument('--verbose', type=str,
                        default=True,
                        help='Enable outputs to terminal')
    parser.add_argument('--mode', type=str,
                        default='quantitative',
                        choices=['quantitative', 'qualitative'],
                        help='Mode of evaluation: `quantitative` (running on OOD dataset) or `qualitative` '
                             '(running on unknown dataset (containing ID and OOD data))')
    parser.add_argument('--retention_times', type=str,
                        help='Use retention times from GCxGC as input features',
                        default=0)
    parser.add_argument('--fn_detection_rate', type=float,
                        help='Accepted False Negative rate for OOD detection',
                        default=0.95)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available!")

    # Define save paths
    suffix = "_w_rt" if args.retention_times else ""
    args.id_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}"
    args.id_train_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}"
    args.ood_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}{suffix}/{args.ood_dataset_name}"
    args.log_dir_path = f"./logs/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}/{args.ood_dataset_name}"
    args.detector_save_dir_path = f"{args.save_root_path}/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}/ood_detectors"
    args.figures_save_dir_path = f"./figures/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}/{args.ood_dataset_name}"
    args.trained_models_save_dir_path = f"./trained_models/seed-{args.seed}/{args.model_name}{suffix}/{args.id_dataset_name}"

    # Create necessary folders
    mkdir(args.log_dir_path)
    mkdir(args.id_save_dir_path)
    mkdir(args.id_train_save_dir_path)
    mkdir(args.ood_save_dir_path)
    mkdir(args.detector_save_dir_path)
    mkdir(args.figures_save_dir_path)
    mkdir(args.trained_models_save_dir_path)
    mkdir('./trained_models')
    mkdir('./figures')
    mkdir('./predictions')

    # Load classification_models config
    args.model_parameters = import_yaml_config(f'configs/classification_models/{args.model_name}.yaml')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, ood_detector_name):
    # Define save paths
    save_dir_paths = {}
    save_dir_paths['id_test'] = args.id_save_dir_path
    save_dir_paths['id_train'] = args.id_train_save_dir_path
    save_dir_paths['ood'] = args.ood_save_dir_path

    # Initialize and load/train model + load ID and OOD data
    model = create_model(args.model_name)
    model.set_model(args)

    # Run model on ID and OOD data (or load model outputs from file) to produce model outputs (feas, logits, labels)
    model_outputs = {}
    try:
        for fold in ['id_train', 'id_test', 'ood']:
            model_outputs[fold] = load_object(f"{save_dir_paths[fold]}/model_outputs_{fold}.pkl")
    except:
        model_outputs = model.get_model_outputs()
        for fold in model_outputs.keys():
            save_object(model_outputs[fold], f"{save_dir_paths[fold]}/model_outputs_{fold}.pkl")

    # Get ID classification performance
    id_classification_perf = model.get_id_classification_scores()

    # Run OOD ood_detectors
    print(f"[INFO] Running {ood_detector_name} on {args.model_name} model - ID dataset {args.id_dataset_name}, OOD dataset {args.ood_dataset_name}.")

    # Initialize OOD detector
    saved_detector_path = f"{args.detector_save_dir_path}/{ood_detector_name}.pt"
    try:
        ood_detector = load_object(saved_detector_path)
    except:
        ood_detector = create_ood_detector(ood_detector_name)
        ood_detector.setup(args, model_outputs['id_train'])
        save_object(ood_detector, saved_detector_path)

    # Get OOD scores for ID and OOD data
    print(f"[INFO] Generate OOD scores and labels.")
    id_scores = ood_detector.infer(model_outputs['id_test'])
    ood_scores = ood_detector.infer(model_outputs['ood'])
    scores = torch.cat([id_scores, ood_scores], dim=0).numpy()
    if args.mode == 'qualitative':
        classification_predictions = model_outputs['ood']['decision_scores'].argmax(1)
        return scores, classification_predictions, id_classification_perf

    # Visualize score distributions
    scores_ = {"id": {ood_detector_name: id_scores}, "ood": {ood_detector_name: ood_scores}}
    plot_distributions(scores_, ood_detector_name, save_plot_dir=args.figures_save_dir_path, nbins=300,
                       use_logspace=args.detector_parameters['plot_distribution_logscale'],
                       normalize_density=args.detector_parameters['plot_distribution_norm'], save_plot=True)
    detection_labels_pred = np.concatenate([np.ones_like(model_outputs['id_test']['labels_true'], dtype=np.int64),
                                            np.zeros(model_outputs['ood']['logits'].shape[0], dtype=np.int64)])
    return scores, detection_labels_pred, id_classification_perf


def main():
    args = get_args()
    set_seed(args.seed)

    scores_set = {}
    class_performance = {}
    labels = {}
    for ood_detector in args.ood_detectors:
        print("=====================================================================")
        print(f"[INFO] Evaluating {ood_detector} detector.")
        args.detector_parameters = import_yaml_config(f"configs/ood_detectors/{ood_detector}.yaml")
        scores_set[ood_detector], labels[ood_detector], class_performance[ood_detector] = evaluate(args, ood_detector)

    if args.mode == 'quantitative':
        evaluation_metrics = get_performance(scores_set, labels[args.ood_detectors[0]], class_performance)
        save_performance(evaluation_metrics, args, f"{args.log_dir_path}/ood-{args.ood_dataset_name}.csv")
    elif args.mode == 'qualitative':
        # save only once
        save_object(class_performance[args.ood_detectors[0]], f"{args.ood_save_dir_path}/classification_perf_id.pt")
        save_object(labels[args.ood_detectors[0]], f"{args.ood_save_dir_path}/model_predictions_ood.pt")
        for ood_detector in args.ood_detectors:
            num_ood_samples = len(labels[args.ood_detectors[0]])
            num_id_samples = len(scores_set[ood_detector]) - num_ood_samples
            id_scores = scores_set[ood_detector][:num_id_samples]
            ood_scores = scores_set[ood_detector][num_id_samples:]
            save_object(id_scores, f"{args.ood_save_dir_path}/{ood_detector}_scores_id.pt")
            save_object(ood_scores, f"{args.ood_save_dir_path}/{ood_detector}_scores_ood.pt")
        save_performance_and_visualize(scores_set, labels[args.ood_detectors[0]],
                                       class_performance[args.ood_detectors[0]], f"{args.log_dir_path}/ood-{args.ood_dataset_name}.csv", args)

if __name__ == '__main__':
    main()
