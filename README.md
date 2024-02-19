
# Introduction
This is a repository for a Master Thesis at the Czech Technical University in Prague, Faculty of Electrical Engineering.

Author: Pavel Linder

## :ledger: Index

- [About](#beginner-about)
- [Usage](#zap-usage)
- [Development](#wrench-development)
  - [Pre-Requisites](#notebook-pre-requisites)
  - [Development Environment](#nut_and_bolt-development-environment)
  - [File Structure](#file_folder-file-structure)

##  :beginner: About
The topic of the thesis is Out-Of-Distribution (OOD) detection GCxGC data classification.
There are two classification models used for this task: CNN and SVM.
They are both trained from scratch and fine-tuned with GCxGC data.
Details can be found in the notebooks `notebooks/` folder.

The thesis in pdf format can be found in `Out_Of_Distribution_Detection_for_GCxGC_Data_Classification.pdf`.

The OOD detection is done using these OOD detectors:
- MSP
- Energy
- KL
- MaxLogit
- Mahalanobis
- DeepNeighbours
- SSD
- NNGuide
  
Evaluation is done using a custom private dataset. The dataset is not publicly available.

Evaluation pipeline is executed from the `main.py` file. It computes the metrics and saves the results to the `logs/` folder.
There are two modes of evaluation:
- Quantitative experiment (producing metrics)
- Qualitative experiment (producing visualizations)
You can switch between the two using arguments in the `main.py` file.

OOD detection metrics are as follows:
- AUROC, AUIN, AUOUT, FPR95

Classification metrics are as follows:
- Accuracy, Average Precision, Average Recall

## :zap: Usage
Add src/ to Python path via Command line:
```
$ export PYTHONPATH="${PYTHONPATH}:./src/"
```
or inside a Python function:
```
import sys
sys.path.append('./src')
```
Run the main.py file with arguments:
```
$ python3 main.py <ARGUMENTS>
```
Arguments are as follows:
- `--model_name` - name of the classification model (cnn, svm)
- `--ood_detector_name` - name of the OOD detector (energy, msp, kl, maxlogit, mahalanobis, knn, ssd, nnguide)
- `--id_dataset_name` - name of the ID dataset (spectrum_exported)
- `--ood_dataset_name` - name of the OOD dataset (gaussian)
- `--ood_detectors` - list of OOD detectors to use (energy, msp)
- `--num_classes` - number of classes in the ID dataset (70)
- `--k_fold_number` - number of folds for cross-validation (5)
- `seed` - seed for reproducibility (1)
- `data_root_path` - path to the data folder (./data/)
- `save_root_path` - path where models outputs will be saved (./saved_model_outputs/)
- `verbose` - verbosity (True)
- `mode` - mode of the evaluation pipeline (quantitative, qualitative)
- `retention_times` - using retention times from GCxGC chromatography as additional features (False)
- `fn_detection_rate` - accepted False Negative rate for OOD detection

##  :wrench: Development
### :notebook: Pre-Requisites
- MiniConda3
- Pip
- Python 3.10

### :nut_and_bolt: Development Environment

1. Clone this repo
2. Activate the conda environment

```
$ conda env create --name OOD_for_GCxGC_classifier --file=environment.yml
```

3. Activate the environment

```
$ conda activate OOD_for_GCxGC_classifier
```

### :file_folder: File Structure
```
.
├── configs/
│   ├── classification_models/
│   │   ├── cnn.yaml
│   │   └── ...
│   ├── ood_detectors/
│   │   └── energy.yaml
│   │   └── ...
├── data/
│   ├── id/
│   │   ├── custom in distibution dataset/
│   │   │   ├── dataset files
│   ├── ood/
│   │   ├── custom out of distribution dataset/
│   │   │   ├── dataset files
├── figures/ 
├── logs/
├── notebooks/
└── saved_model_outputs/
│   ├── seed-1/
│   │   ├── ood_detector_name/
│   │   │   ├── custom in distibution dataset/
│   │   │   │   ├── detectors/energy.pt
│   │   │   │   ├── detector/...
│   │   │   │   ├── model_outputs_id.pt
│   │   │   ├── custom out of distribution dataset/
│   │   │   │   ├── model_outputs_ood.pt
│   ├── seed-2/
├── src/
├── trained_models/
├── environment.yml
├── main.py
├── Out_Of_Distribution_Detection_for_GCxGC_Data_Classification.pdf
├── requirements.txt

```


| File/Folder Name | Description                                                         |
|------------------|---------------------------------------------------------------------|
| configs/         | Parameters for classification models and OOD detectors              |
| data/            | Data folder (insert your downloaded data here)                      |
| figures/         | Visualizations                                                      |
| logs/            | Evaluation metrics as csv files  (follow same structure as outputs) |
| notebooks/       | Jupyter notebooks                                                   |
| saved_model_outputs/ | Saved model outputs (logits, features)                              |
| src/             | Source code                                                         |
| trained_models/  | Trained weights for the models                                      |
| environment.yml  | Conda environment file                                              |
| main.py          | Main file for running the experiments                               |
| requirements.txt | Pip requirements file                                               |
