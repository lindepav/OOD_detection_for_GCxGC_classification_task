from interface import ClassificationModel
from sklearn.svm import SVC
from utils import load_dataset, save_object, load_object, probs_to_logits, convert_categorical_labels
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
from utils import subsample, subset_dataset_from_indices
from sklearn.preprocessing import Normalizer


class SVMModel(ClassificationModel):
    def set_model(self, args):
        super().set_model(args)
        self._data = {}
        for fold in self._folds:
            if fold == 'id':
                data_dict = self.load_id_data()
            elif fold == 'ood':
                data_dict = self.load_ood_data()
            else:
                data_dict = None
            self._data[fold] = data_dict
        self.train_subsample_ratio = self._model_parameters['train_subsample_ratio']

        # Check if we have trained model, if not, train it, if yes, load it
        suffix = "_w_rt" if self._retention_times else ""
        try:
            self._model = load_object(f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}.pt")
            train_ids = load_object(f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_train_data_indices.pt")
            test_ids = load_object(f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_test_data_indices.pt")
            self._id_score = load_object(f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_scores.pt")
            self._data['id_train'] = subset_dataset_from_indices(self._data['id'], train_ids)
            self._data['id_test'] = subset_dataset_from_indices(self._data['id'], test_ids)
            print(f"[INFO] Pretrained SVM loaded from {self._trained_models_save_dir_path}/{args.model_name}{suffix}.pt")
            print(f"[INFO] ID train and ID test data loaded.")
        except:
            print(f"[INFO] No pretrained SVM found, training new model...")
            self.train_k_folds()
            self._data['id_train'] = subset_dataset_from_indices(self._data['id'], self._train_ids)
            self._data['id_test'] = subset_dataset_from_indices(self._data['id'], self._test_ids)
            save_object(self._model, f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}.pt")
            save_object(self._id_score, f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_scores.pt")
            save_object(self._train_ids, f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_train_data_indices.pt")
            save_object(self._test_ids, f"{self._trained_models_save_dir_path}/{args.model_name}{suffix}_id_test_data_indices.pt")
            print(f"[INFO] Training finished.")
            print(f"[INFO] ID train and ID test data loaded.")
        if args.mode == 'qualitative':
            self.train_subsample_ratio = 1.0
        if self.train_subsample_ratio < 1.:
            subsample(self._data['id_train'], alpha=self.train_subsample_ratio)

    def preprocess_data(self, data):
        normalizer = Normalizer(norm='max')
        if self._retention_times:
            data['data_points'][:, :-2] = normalizer.fit_transform(data['data_points'][:, :-2])
            data['data_points'][:, -2] /= 3600   # retention time 1
            data['data_points'][:, -1] /= 10     # retention time 2
        else:
            data['data_points'] = normalizer.fit_transform(data['data_points'])
        return data

    def load_id_data(self):
        if 'synthetic_spectrum' in self._id_dataset_name:
            id_dataset_name = "_".join(self._id_dataset_name.split('_')[:-2])
            if self._retention_times:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=id_dataset_name, use_retention_time=True,
                    verbose=self._verbose)
            else:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=id_dataset_name, verbose=self._verbose)
            data = self.preprocess_data(data)
            id_labels = load_object(self._data_root_path + '/id/' + id_dataset_name +
                                    '/' + self._id_dataset_name + '.pkl')
            id_indices = np.isin(data['data_labels'], id_labels)
            data['data_points'] = data['data_points'][id_indices]
            data['data_labels_orig'] = data['data_labels'][id_indices]
            data['metadata'] = data['metadata'][id_indices]
            # convert categorical labels to numerical again
            data['data_labels'] = convert_categorical_labels(data['data_labels_orig'], data['data_labels_orig'], str_labels=False)
        else:
            if self._retention_times:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=self._id_dataset_name, use_retention_time=True, verbose=self._verbose)
            else:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=self._id_dataset_name, verbose=self._verbose)
            data = self.preprocess_data(data)
        return data
    def load_ood_data(self):
        start = time.time()

        if 'synthetic_spectrum' in self._ood_dataset_name:
            id_dataset_name = "_".join(self._id_dataset_name.split('_')[:-2])
            if self._retention_times:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=id_dataset_name, use_retention_time=True,
                    verbose=self._verbose)
            else:
                data = load_dataset(
                    data_dir=self._data_root_path + '/id/', dataset_name=id_dataset_name, verbose=self._verbose)
            ood_labels = load_object(self._data_root_path + '/ood/' + "_".join(self._ood_dataset_name.split('_')[:-2]) +
                                    '/' + self._ood_dataset_name + '.pkl')
            ood_indices = np.isin(data['data_labels'], ood_labels)
            data['data_points'] = data['data_points'][ood_indices]
            data['data_labels_orig'] = data['data_labels'][ood_indices]
            data['metadata'] = data['metadata'][ood_indices]
        else:
            data = {'data_points': [], 'data_labels': [], 'metadata': []}
            if self._retention_times:
                data['data_points'] = load_object(self._data_root_path + '/ood/' + self._ood_dataset_name + '/' +
                                                  self._ood_dataset_name + '_rt.pkl')
            else:
                data['data_points'] = load_object(self._data_root_path + '/ood/' + self._ood_dataset_name + '/' +
                               self._ood_dataset_name + '.pkl')

        # we have 70 classes (starting from 0), so we assign 70 to OOD
        data['data_labels'] = np.ones(len(data['data_points'])) * 70

        print(f"[INFO] Dataset { self._ood_dataset_name} loaded loaded in {(time.time() - start):.1f} seconds.")
        print(f"[INFO] Input data shapes: X[{data['data_points'].shape}], y[{data['data_points'].shape}]")
        return self.preprocess_data(data)

    def train(self, model, train_data, test_data):
        X_train, y_train = train_data
        X_test, y_test = test_data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred, average=None).mean()
        recall = recall_score(y_test, y_pred, average=None).mean()
        performance = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }
        return model, performance
    def train_k_folds(self):
        start = time.time()
        X = self._data['id']['data_points']
        y = self._data['id']['data_labels']
        results = {
            "accuracy": [],
            "precision": [],
            "recall": []
        }
        best_score = 0
        kfold = StratifiedKFold(n_splits=self._k_fold_number, shuffle=True, random_state=self._seed)
        with tqdm(kfold.split(X, y), unit=" fold") as k_fold:
            for fold, (train_ids, test_ids) in enumerate(k_fold):
                k_fold.set_description(f"Fold {fold + 1}")
                X_train = X[train_ids]
                y_train = y[train_ids]
                X_test = X[test_ids]
                y_test = y[test_ids]
                model = SVC(kernel=self._model_parameters['kernel'],
                            decision_function_shape='ovr',
                            probability=True,
                            shrinking=True,
                            break_ties=True,  # break ties according to the confidence values of decision_function
                            class_weight='balanced',  # class imbalances by class weights
                            gamma=self._model_parameters['gamma'],
                            C=self._model_parameters['C'],
                            random_state=self._seed,
                            verbose=0)
                model, performance = self.train(model, [X_train, y_train], [X_test, y_test])
                results['accuracy'] += [performance['accuracy']]
                results['precision'] += [performance['precision']]
                results['recall'] += [performance['recall']]
                if performance['accuracy'] > best_score:
                    best_score = performance['accuracy']
                    self._model = model
                    self._train_ids = train_ids
                    self._test_ids = test_ids
                k_fold.set_postfix(accuracy=100. * performance['accuracy'],
                                   precision=100. * performance['precision'], recall=100. * performance['recall'])
        self._id_score = {key: float(np.mean(values)) for key, values in results.items()}
        print("================================================")
        print(f"[INFO] total time taken to train the model (using {self._k_fold_number}-fold evaluation: {(time.time() - start):.2f}s "
              f"or {(time.time() - start) / 60:.2f}min.")
        print(f"Results for ID dataset {self._id_dataset_name}:")
        print(f"Accuracy: {100.*self._id_score['accuracy']:.2f}%, Precision: {100.*self._id_score['precision']:.2f}%,"
              f" Recall: {100.*self._id_score['recall']:.2f}%")
        print("================================================")


    def extract_features(self, data):
        start = time.time()
        print(f"[INFO] Extracting features from {data.shape[0]} data points.")
        # computes the signed distance of a point from the boundary
        scores = self._model.decision_function(data)
        probs = self._model.predict_proba(data)
        probs = torch.from_numpy(probs)
        logits = probs_to_logits(probs)
        out = {}
        out['logits'] = logits
        out['decision_scores'] = scores
        print(f"[INFO] Features extracted in {(time.time() - start)//60} min {(time.time() - start):.0f}s")
        return out

    def get_model_outputs(self):
        model_outputs = {}
        for fold in ['id_train', 'id_test', 'ood']:
            model_outputs[fold] = {}
            out = self.extract_features(self._data[fold]['data_points'])
            model_outputs[fold]["feas"] = self._data[fold]['data_points']
            model_outputs[fold]["logits"] = out["logits"]
            model_outputs[fold]["labels_true"] = self._data[fold]["data_labels"]
            model_outputs[fold]["decision_scores"] = out["decision_scores"]
        return model_outputs

    def predict_from_logits(self, logits):
        return torch.max(logits, dim=-1)[1]


    def get_id_classification_scores(self):
        return self._id_score
