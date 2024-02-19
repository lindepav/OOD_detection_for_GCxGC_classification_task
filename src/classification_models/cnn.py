from interface import ClassificationModel
from utils import load_object, save_object, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Conv1d, MaxPool1d, Softmax, Linear, Flatten, Dropout
from tqdm import tqdm
from utils import subsample, subset_dataset_from_indices
from sklearn.preprocessing import Normalizer


class CNNModel(ClassificationModel):
    def set_model(self, args):
        super().set_model(args)

        # Load model parameters
        self._loss_fn = nn.CrossEntropyLoss()
        self._num_channels = self._model_parameters['num_channels']
        self._num_classes = self._model_parameters['num_classes']
        self._N_EPOCHS = self._model_parameters['n_epochs']
        self._BATCH_SIZE = self._model_parameters['batch_size']
        self._LEARNING_RATE = self._model_parameters['learning_rate']
        self._DROPOUT_RATE = self._model_parameters['dropout_rate']
        self.train_subsample_ratio = self._model_parameters['train_subsample_ratio']

        # Load data from disk
        self._data = {}
        for fold in self._folds:
            if fold == 'id':
                data_dict = self.load_id_data()
            elif fold == 'ood':
                data_dict = self.load_ood_data()
            else:
                data_dict = None
            self._data[fold] = data_dict


        # Check if we have trained model, if not, train it, if yes, load it
        try:
            self._model = SimpleCNN(self._num_channels, self._num_classes, self._DROPOUT_RATE).to(self._device)
            self._model.load_state_dict(torch.load(f"./trained_models/{args.model_name}.md", map_location=self._device))
            train_ids = load_object(f"./trained_models/{args.model_name}_id_train_data_indices.pt")
            test_ids = load_object(f"./trained_models/{args.model_name}_id_test_data_indices.pt")
            self._data['id_train'] = subset_dataset_from_indices(self._data['id'], train_ids)
            self._data['id_test'] = subset_dataset_from_indices(self._data['id'], test_ids)
            self._id_score = load_object(f"./trained_models/{args.model_name}_id_scores.pt")
            print(f"[INFO] Pretrained CNN loaded from ./trained_models/{args.model_name}.pt")
            print(f"[INFO] ID train and ID test data loaded.")
        except:
            print(f"[INFO] No pretrained CNN found, training new model...")
            self.train_k_folds()
            self._data['id_train'] = subset_dataset_from_indices(self._data['id'], self._train_ids)
            self._data['id_test'] = subset_dataset_from_indices(self._data['id'], self._test_ids)
            # TODO: save as one object
            torch.save(self._model.state_dict(), f"./trained_models/{args.model_name}.md")
            save_object(self._id_score, f"./trained_models/{args.model_name}_id_scores.pt")
            save_object(self._train_ids, f"./trained_models/{args.model_name}_id_train_data_indices.pt")
            save_object(self._test_ids, f"./trained_models/{args.model_name}_id_test_data_indices.pt")
            print(f"[INFO] Training finished.")
            print(f"[INFO] ID train and ID test data loaded.")
        if self.train_subsample_ratio < 1.:
            subsample(self._data['id_train'], alpha=self.train_subsample_ratio)


    def preprocess_data(self, data):
        normalizer = Normalizer(norm='max')
        data['data_points'] = normalizer.fit_transform(data['data_points'])
        return data


    def load_id_data(self):
        ds = load_dataset(
            data_dir=self._data_root_path + '/id/', dataset_name=self._id_dataset_name, verbose=self._verbose)
        ds = self.preprocess_data(ds)
        ds['data_points'] = ds['data_points'].reshape(ds['data_points'].shape[0], 1, ds['data_points'].shape[1])
        return ds


    def load_ood_data(self):
        start = time.time()
        data = {'data_points': [], 'data_labels': [], 'metadata': []}
        data['data_points'] = load_object(self._data_root_path + '/ood/' + self._ood_dataset_name + '/' +
                                          self._ood_dataset_name + '.pkl')
        data['data_points'] = data['data_points'].reshape(data['data_points'].shape[0], 1, data['data_points'].shape[1])

        if self._ood_dataset_name == 'M29_9_system2':
            # we don't have labels for this dataset, so we assign nan to all of them
            data['data_labels'] = np.empty(len(data['data_points']))
            data['data_labels'][:] = np.nan
        else:
            # we have 70 classes (starting from 0), so we assign 70 to OOD
            data['data_labels'] = np.ones(len(data['data_points'])) * 70
        print(f"[INFO] Dataset {self._ood_dataset_name} loaded loaded in {(time.time() - start):.1f} seconds.")
        print(f"[INFO] Input data shapes: X[{data['data_points'].shape}], y[{data['data_points'].shape}]")
        return self.preprocess_data(data)

    def train(self, model, train_data, test_data):
        model, optimizer = model
        train_loader = train_data
        validation_loader = test_data
        num_train_samples = len(train_loader.sampler.indices)
        num_val_samples = len(validation_loader.sampler.indices)
        num_train_steps, num_val_steps = num_train_samples // self._BATCH_SIZE, num_val_samples // self._BATCH_SIZE
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        if self._verbose:
            print("[INFO] Training the network...")
            start = time.time()
        iterator_epochs = tqdm(range(self._N_EPOCHS), unit="epoch")
        for epoch in iterator_epochs:
            train_loss = 0
            train_correct = 0
            iterator_batches = train_loader
            iterator_epochs.set_description(f"Epoch {epoch}")
            for batch_n, (xi, yi) in enumerate(iterator_batches):
                xi, yi = xi.to(self._device), yi.to(self._device)
                optimizer.zero_grad()
                z = model(xi)
                loss = self._loss_fn(z, yi)
                loss.backward()
                optimizer.step()

                train_loss += loss
                correct = ((z.argmax(1) == yi).type(torch.float).sum().item())
                train_correct += correct

            epoch_train_loss = (train_loss / num_train_steps).item()
            epoch_train_correct = train_correct / num_train_samples
            H["train_loss"].append(epoch_train_loss)
            H["train_acc"].append(epoch_train_correct)
            iterator_epochs.set_postfix(loss=epoch_train_loss, accuracy=100. * epoch_train_correct)
        if self._verbose:
            print(
                f"[INFO] Total time taken to train the model: {(time.time() - start):.2f}s or {(time.time() - start) / 60:.2f}min.")

        # Evaluate on validation set
        val_correct = 0
        val_loss = 0
        predictions = []
        labels = []
        model.eval()
        with torch.no_grad():
            for xi, yi in validation_loader:
                xi, yi = xi.to(self._device), yi.to(self._device)
                z = model(xi)
                y_pred = z.argmax(1).cpu().numpy()
                predictions.append(y_pred)
                labels.append(yi.cpu().numpy())
                val_loss += self._loss_fn(z, yi)
                val_correct += ((z.argmax(1) == yi).type(torch.float).sum().item())
        val_loss = (val_loss / num_val_steps).item()
        val_correct = val_correct / num_val_samples
        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)
        H['test_precision'] = precision_score(y_true=labels, y_pred=predictions, average='micro', zero_division='warn')
        H['test_recall'] = recall_score(y_true=labels, y_pred=predictions, average='micro', zero_division='warn')
        H["test_loss"] = val_loss
        H["test_acc"] = val_correct
        return model, H, predictions, labels


    def train_k_folds(self):
        start = time.time()
        results = {
            "accuracy": [],
            "precision": [],
            "recall": []
        }
        best_score = 0
        X = self._data['id']['data_points']
        y = self._data['id']['data_labels']
        whole_dataset = Data(X, y)
        kfold = StratifiedKFold(n_splits=self._k_fold_number, shuffle=True, random_state=self._seed)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(X, y)):
            print(f"[INFO] Fold [{fold + 1}/{self._k_fold_number}]: Starting.")
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            trainloader = DataLoader(
                dataset=whole_dataset,
                batch_size=self._BATCH_SIZE,
                sampler=train_subsampler)
            testloader = DataLoader(
                dataset=whole_dataset,
                batch_size=self._BATCH_SIZE,
                sampler=test_subsampler)
            model = SimpleCNN(self._num_channels, self._num_classes, self._DROPOUT_RATE).to(self._device)
            optimizer = Adam(model.parameters(), lr=self._LEARNING_RATE)

            trained_model, performance, y_pred, y_true = self.train(
                                                            model=[model, optimizer],
                                                            train_data=trainloader,
                                                            test_data=testloader)
            if set(y_true) - set(y_pred):
                print(f"[WARNING] Computation of some metrics may be incorrect. "
                      f"\n set(y_true) - set(y_pred) = {set(y_true) - set(y_pred)}")
            print('[INFO] Training process has finished.')

            # Evaluation for this fold
            print(f'Accuracy for fold #{fold + 1}: {(100. * performance["test_acc"]):.2f}%')
            print(f'Precision for fold #{fold + 1}: {(100. * performance["test_precision"]):.2f}%')
            print(f'Recall for fold #{fold + 1}: {(100. * performance["test_recall"]):.2f}%')
            print(f'Loss for fold #{fold + 1}: {performance["test_loss"]:.2f}')
            print("================================================")
            results['accuracy'] += [performance['test_acc']]
            results['precision'] += [performance['test_precision']]
            results['recall'] += [performance['test_recall']]
            if performance['test_acc'] > best_score:
                best_score = performance['test_acc']
                self._model = trained_model
                self._train_ids = train_ids
                self._test_ids = test_ids

        self._id_score = {key: float(np.mean(values)) for key, values in results.items()}
        print("================================================")
        print(
            f"[INFO] total time taken to train the model (using {self._k_fold_number}-fold evaluation: {(time.time() - start):.2f}s "
            f"or {(time.time() - start) / 60:.2f}min.")
        print(f"Results for ID dataset {self._id_dataset_name}:")
        print(f"Accuracy: {100. * self._id_score['accuracy']:.2f}%, Precision: {100. * self._id_score['precision']:.2f}%,"
              f" Recall: {100. * self._id_score['recall']:.2f}%")
        print("================================================")


    def extract_features(self, data):
        print(f"[INFO] Extracting features from {len(data.dataset)} data points.")
        self._model.to(self._device)
        self._model.eval()
        features = [[]] * len(data)
        logits = [[]] * len(data)
        labels = [[]] * len(data)

        network_layers_outputs = {}

        # using hooks to get intermediate features
        # ref. https://kozodoi.me/blog/20210527/extracting-features#2.-Why-do-we-need-intermediate-features?
        def get_features(name):
            def hook(model, input, output):
                network_layers_outputs[name] = output.detach()

            return hook
        self._model.dense2.register_forward_hook(get_features('features'))

        for i, (xi, yi) in tqdm(enumerate(data), unit=f' / {len(data)} batch', desc="Extracting features"):
            xi, yi = xi.to(self._device), yi.to(self._device)
            with torch.no_grad():
                _logits = self._model(xi)
            _raw_feats = network_layers_outputs['features'].cpu()
            _feats = F.normalize(_raw_feats, dim=1)

            features[i] = _feats
            logits[i] = _logits.cpu()
            labels[i] = yi.cpu()

        features = torch.cat(features, dim=0)
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        print(f"[INFO] Successfully extracted features")

        return {"feas": features, "logits": logits, "labels": labels}


    def get_model_outputs(self):
        model_outputs = {}
        for fold in ['id_train', 'id_test', 'ood']:
            model_outputs[fold] = {}
            dataset = Data(self._data[fold]['data_points'], self._data[fold]['data_labels'])
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self._BATCH_SIZE)
            out = self.extract_features(data_loader)
            model_outputs[fold]["logits"] = out["logits"]
            model_outputs[fold]["feas"] = out["feas"]
            model_outputs[fold]["labels_true"] = out["labels"]

        return model_outputs

    def get_id_classification_scores(self):
        return self._id_score

    def predict_from_logits(self, logits):
        return torch.max(logits, dim=-1)[1]


class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_classes, dropout_rate):
        super(SimpleCNN, self).__init__()

        self.conv1 = Conv1d(in_channels=num_channels, out_channels=32,
                            kernel_size=5, padding=2, stride=1)
        self.maxpool = MaxPool1d(kernel_size=5, stride=2, padding=1)
        self.conv2 = Conv1d(in_channels=32, out_channels=64,
                            kernel_size=5, padding=2, stride=1)
        self.drop1 = Dropout(dropout_rate)
        self.dense1 = Linear(in_features=400, out_features=100)
        self.flatten = Flatten()
        self.drop2 = Dropout(dropout_rate)
        self.dense2 = Linear(in_features=6400, out_features=500)
        self.output = Linear(in_features=500, out_features=num_classes)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.flatten(x)
        x = self.drop2(x)
        x = self.dense2(x)
        x = self.output(x)
        return x


class Data(Dataset):
    def __init__(self, x_train, y_train):
        self.x = torch.from_numpy(x_train)
        self.y = torch.from_numpy(y_train)
        self.len = self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len