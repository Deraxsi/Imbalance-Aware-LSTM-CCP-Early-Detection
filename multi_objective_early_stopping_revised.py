import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa: E402

import glob

import time

import json

import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.optimizers import Nadam

from tensorflow.keras import Model, Input

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, RNN

from tensorflow_addons.rnn import LayerNormLSTMCell

import keras_tuner as kt


class NumpyArrayEncoder(json.JSONEncoder):
    """This class helps us to write numpy ndarray into json-like text files"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class Specificity(tf.keras.metrics.Metric):
    """This class provides the specificity metric"""

    def __init__(self, name='specificity'):
        """
        Args:
            name (str): denotes the name of metric
        """
        super(Specificity, self).__init__(name=name)
        self.specificity = self.add_weight(name='spec', initializer='zeros')

    def reset_state(self):
        """this method resets the specificity variable to zero in each epoch"""
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """This function updates the value of specificity in each epoch

        Args:
            y_true (Tensor of type float): actual labels of size (batch_size, number of classes)
            y_pred (Tensor of type float): predicted labels of size (batch_size, number of classes)
            sample_weight (Tensor of type float): None

        Returns: None

        """
        bool_labels = tf.cast(y_true, tf.bool)
        bool_predicted_labels = tf.vectorized_map(lambda p: p > 0.5, y_pred)
        reshaped_labels = tf.reshape(bool_labels, [-1])
        reshaped_predicted_labels = tf.reshape(bool_predicted_labels, [-1])
        cm = tf.math.confusion_matrix(labels=reshaped_labels, predictions=reshaped_predicted_labels,
                                      dtype=tf.float32, num_classes=2)

        no_truely_classified = tf.linalg.diag_part(cm)
        no_samples_per_class = tf.reduce_sum(cm, axis=1)
        true_rates = no_truely_classified / (no_samples_per_class + tf.constant(1e-12))

        self.specificity.assign(true_rates[0])

    def result(self):
        """provides calculated specificity value"""
        return self.specificity


class Sensitivity(tf.keras.metrics.Metric):
    """This class provides the Sensitivity metric"""

    def __init__(self, name='sensitivity'):
        """
        Args:
            name (str): denostes the name of metric
        """
        super(Sensitivity, self).__init__(name=name)
        self.sensitivity = self.add_weight(name='sens', initializer='zeros')

    def reset_state(self):
        """this method resets the sensitivity variable to zero in each epoch"""
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """This function updates the value of sensitivity in each epoch

        Args:
            y_true (Tensor of type float): actual labels of size (batch_size, number of classes)
            y_pred (Tensor of type float): predicted labels of size (batch_size, number of classes)
            sample_weight (Tensor of type float): corresponding weights of the labels, defined by the
                                                  class_weight={0:majors_weight, 1:minors_weights}

        Returns: None

        """
        bool_labels = tf.cast(y_true, tf.bool)
        bool_predicted_labels = tf.vectorized_map(lambda p: p > 0.5, y_pred)
        reshaped_labels = tf.reshape(bool_labels, [-1])
        reshaped_predicted_labels = tf.reshape(bool_predicted_labels, [-1])
        cm = tf.math.confusion_matrix(labels=reshaped_labels, predictions=reshaped_predicted_labels,
                                      dtype=tf.float32, num_classes=2)
        no_truely_classified = tf.linalg.diag_part(cm)
        no_samples_per_class = tf.reduce_sum(cm, axis=1)
        true_rates = no_truely_classified / (no_samples_per_class + tf.constant(1e-12))

        self.sensitivity.assign(true_rates[1])

    def result(self):
        """provides calculated sensitivity value"""
        return self.sensitivity


class GeometricMean(tf.keras.metrics.Metric):
    """This class provides the G-mean metric, which can be used for model selection"""

    def __init__(self, name='geometric_mean', no_classes=2, **kwargs):
        """
        Args:
            name (str): the metric name
            no_classes (int): the number of classes
            **kwargs (dict): dictionary of extra parameters

        """
        super(GeometricMean, self).__init__(name=name, **kwargs)
        self.no_classes = no_classes
        self.gmean = self.add_weight(name='gm', initializer='zeros')

    def reset_state(self):
        """this function resets the gmean variable to zero in each epoch"""
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """This function updates the value of gmean in each epoch

        Args:
            y_true (Tensor of type float): actual labels of size (batch_size, number of classes)
            y_pred (Tensor of type float): predicted labels of size (batch_size, number of classes)
            sample_weight (Tensor of type float): corresponding weights of the labels, defined by the
                                                  class_weight={0:majors_weight, 1:minors_weights}

        Returns: None

        """
        self.gmean.assign(self.gmean_caculator(y_true, y_pred))

    def gmean_caculator(self, y_true, y_pred):
        """This calculates the G-mean indicator,

        Args:
            y_true (Tensor of type float): actual labels of size (batch_size, number of classes)
            y_pred (Tensor of type float): predicted labels of size (batch_size, number of classes)

        Returns:
            gmean value (Tensor of size 1 and type float)

        """
        bool_labels = tf.cast(y_true, tf.bool)
        bool_predicted_labels = tf.vectorized_map(lambda p: p > 0.5, y_pred)
        reshaped_labels = tf.reshape(bool_labels, [-1])
        reshaped_predicted_labels = tf.reshape(bool_predicted_labels, [-1])
        cm = tf.math.confusion_matrix(labels=reshaped_labels, predictions=reshaped_predicted_labels,
                                      dtype=tf.float32, num_classes=2)

        # print(cm)

        no_truely_classified = tf.linalg.diag_part(cm)
        no_samples_per_class = tf.reduce_sum(cm, axis=1)
        gmean = tf.sqrt(tf.reduce_prod(no_truely_classified / (no_samples_per_class + tf.constant(1e-12))))

        return gmean

    def result(self):
        """Returns: the calculated gmean value though the train_step and test_step modules"""
        return self.gmean


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, label_smoothing):
        super(FocalLoss, self).__init__(reduction='sum')
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def __call__(self, y_true, y_pred, sample_weight=None):

        if self.label_smoothing:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        cross_entropy = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        cross_entropy = tf.reshape(cross_entropy, (-1, 1))
        if sample_weight is not None:
            cross_entropy = tf.multiply(cross_entropy, sample_weight)

        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        modulating_factor = tf.pow((1.0 - p_t), self.gamma)
        loss_vector = modulating_factor * cross_entropy
        return loss_vector


class Classifiers:
    """This class provides lstm classifiers for heatmap calculation"""

    def __init__(self, hp, best_strategy, imbalanced_ratio, pattern_name, subclass_weights, minimum_no_epochs,
                 maximum_no_epochs, patience, kappas):
        """
        Args:
            hp (dict): best hyperparameters.
        """
        self.hp = hp

        self.is_stratified_sampling = "Stratified" in best_strategy
        self.is_cost_sensitive = "cost-sensitive" in best_strategy
        if self.is_cost_sensitive:
            self.is_equitable = not ("inequitable" in best_strategy)
        else:
            self.is_equitable = None

        self.imbalanced_ratio = imbalanced_ratio
        self.pattern_name = pattern_name
        self.subclass_weights = subclass_weights

        self.minimum_no_epochs = minimum_no_epochs
        self.maximum_no_epochs = maximum_no_epochs
        self.patience = patience
        self.kappas = kappas

        self.metric_names = ['loss', 'specificity', 'sensitivity', 'accuracy', 'geometric_mean']

    def build(self, time_window):
        """
        Args:
            hp (dict): dictionary containing information of best model's architecture.
            time_window(int): the value of time_window with which the classifier should be constructed accordingly.
        """
        input_layer = Input(shape=(time_window, 1))
        no_layers = self.hp["no_layers"]
        x = input_layer
        for layer in range(1, no_layers + 1):
            shared_names = f'_in_layer_{layer}'

            no_units = self.hp[f'time_window={time_window}/no_units' + shared_names]
            dropout = self.hp[f'time_window={time_window}/dropout' + shared_names]

            x = Bidirectional(
                LSTM(units=no_units,
                     activation='tanh',
                     dropout=dropout,
                     recurrent_activation='sigmoid',
                     recurrent_dropout=0,
                     unroll=False,
                     use_bias=True,
                     return_sequences=True if layer < no_layers else False))(x)

        output_layer = Dense(units=1, activation=tf.nn.sigmoid)(x)

        model = Model(input_layer, output_layer)

        learning_rate = self.hp['learning_rate']
        optimizer = Nadam(learning_rate, clipvalue=0.5, clipnorm=1.0)

        metrics = [
            Specificity(),
            Sensitivity(),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            GeometricMean()
        ]

        loss = FocalLoss(gamma=2, label_smoothing=0)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)

        return model

    def valid_and_test_data_provider(self, datapoints, datapoints_sizes):
        names = list(datapoints.keys())
        features = {name: np.reshape(datapoints[name][0], (datapoints_sizes[name], -1, 1))
                    for name in names}
        labels = {name: np.reshape(np.ceil(datapoints[name][1]), (datapoints_sizes[name], -1))
                  for name in names}

        weights = {}
        imbalanced_ratio = datapoints_sizes['normal'] / sum(datapoints_sizes.values())

        for name in names:
            if name == "normal":
                if self.is_equitable:
                    weights[name] = np.array([1 / imbalanced_ratio] * datapoints_sizes[name])
                else:
                    total_inequity = (1 / (1 - imbalanced_ratio)) * \
                                     sum([datapoints_sizes[name] * (1 - self.subclass_weights[name])
                                          for name in names if name != 'normal'])
                    weights[name] = np.array([1 / imbalanced_ratio -
                                              total_inequity / datapoints_sizes[name]] * datapoints_sizes[name])
            else:
                if self.is_equitable:
                    weights[name] = np.array([1 / (1 - imbalanced_ratio)] * datapoints_sizes[name])
                else:
                    weights[name] = np.array([self.subclass_weights[name] / (1 - imbalanced_ratio)] *
                                             datapoints_sizes[name])

            weights[name] = np.reshape(weights[name], newshape=(-1, 1))

        x_batch = np.concatenate(list(features.values()), axis=0)
        y_batch = np.concatenate(list(labels.values()), axis=0)
        sw_batch = np.concatenate(list(weights.values()), axis=0)

        if not self.is_cost_sensitive:
            sw_batch = None

        valid_data = x_batch, y_batch, sw_batch
        valid_batch_size = sum(datapoints_sizes.values())
        return valid_data, valid_batch_size

    def learning_curve_visualizer(self, history, time_param):

        fig, axes = plt.subplots(2, 1, sharex=True, squeeze=False)

        axes[(0, 0)].plot(history.history['loss'], color='k', label='train loss', linewidth=1.5)
        axes[(0, 0)].plot(history.history['val_loss'], color='b', label='valid loss')
        axes[(0, 0)].legend(loc='upper center', shadow=True, fontsize='x-small')
        axes[(0, 0)].set(ylabel='loss', title='learning curve')
        axes[(0, 0)].grid()

        axes[(1, 0)].plot(history.history['val_geometric_mean'], color='r', label='valid G-mean')
        axes[(1, 0)].legend(loc='lower right', shadow=True, fontsize='x-small')
        axes[(1, 0)].grid()
        axes[(1, 0)].set(xlabel='epochs', ylabel='G-mean')
        fig.tight_layout()

        curves_paths = os.path.join(os.getcwd(), 'Offline_Learning_Curves', 'In_heatmaps', self.pattern_name)
        curves_save_fname = os.path.join(curves_paths, time_param + '.png')

        if not os.path.exists(curves_paths):
            os.makedirs(curves_paths)

        plt.savefig(curves_save_fname)

        plt.close()

    def fit_per_dataset(self, file_directory):

        with open(file=file_directory, mode='r') as readfile:
            file = json.load(readfile)

        time_window = file["time_window"]
        abnormality_value = file['abnormality_parameters']['parameter_value']

        dataset = file["normalized_dataset"]
        dataset_distribution = file['dataset_distribution']
        train_datapoints = dataset['train']

        valid_data, valid_batch_size = self.valid_and_test_data_provider(dataset['valid'],
                                                                         dataset_distribution['valid'])
        itest_data, itest_batch_size = self.valid_and_test_data_provider(dataset['imbalanced_test'],
                                                                         dataset_distribution['imbalanced_test'])

        btest_data, btest_batch_size = self.valid_and_test_data_provider(dataset['balanced_test'],
                                                                         dataset_distribution['balanced_test'])

        batch_size = self.hp["batch_size"]

        # logdir = os.path.join(os.getcwd(), 'Learning_logs')
        # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        gmean_callback = MultiObjectiveEarlyStopping(patience=self.patience, kappas=self.kappas,
                                                     minimum_no_epochs=self.minimum_no_epochs,
                                                     maximum_no_epochs=self.maximum_no_epochs,
                                                     verbosity=0)

        time_param = f"T={time_window}-prm={abnormality_value}"

        history_callback_dir = os.path.join(os.getcwd(), 'Early_Stopping', self.pattern_name)
        if not os.path.exists(history_callback_dir):
            os.makedirs(history_callback_dir)
        history_fname = os.path.join(history_callback_dir, time_param + '.csv')
        history_callback = tf.keras.callbacks.CSVLogger(history_fname, separator=",", append=False)

        assert time_window == len(train_datapoints['normal'][0][0])
        model = self.build(file["time_window"])

        generator = DataGenerator(dataset=train_datapoints,
                                  class_sizes=dataset_distribution['train'],
                                  batch_size=batch_size,
                                  imbalanced_ratio=self.imbalanced_ratio,
                                  is_stratified_sampling=self.is_stratified_sampling,
                                  is_cost_sensitive=self.is_cost_sensitive,
                                  is_equitable=self.is_equitable,
                                  subclass_weights=self.subclass_weights,
                                  is_shuffling=True)

        history = model.fit(x=generator,
                            epochs=self.maximum_no_epochs,
                            # use_multiprocessing=True,
                            callbacks=[gmean_callback, history_callback],
                            validation_data=valid_data,
                            validation_batch_size=valid_batch_size,
                            verbose=0)

        # self.learning_curve_visualizer(history, time_param)

        trained_classifier = history.model

        outputs = list()

        for method, lstm_weights in gmean_callback.best_weights.items():
            trained_classifier.set_weights(lstm_weights)
            output = [time_window, abnormality_value, method + '_method']

            valid_results = trained_classifier.evaluate(x=valid_data[0],
                                                        y=valid_data[1],
                                                        sample_weight=valid_data[2],
                                                        batch_size=len(valid_data[1]),
                                                        verbose=0)

            imbalanced_test_results = trained_classifier.evaluate(x=itest_data[0],
                                                                  y=itest_data[1],
                                                                  sample_weight=itest_data[2],
                                                                  batch_size=itest_batch_size,
                                                                  verbose=0)

            balanced_test_results = trained_classifier.evaluate(x=btest_data[0],
                                                                y=btest_data[1],
                                                                sample_weight=btest_data[2],
                                                                batch_size=btest_batch_size,
                                                                verbose=0)
            for i in range(len(valid_results)):
                output.extend([valid_results[i], imbalanced_test_results[i], balanced_test_results[i]])
            output.append(gmean_callback.training_times[method])
            outputs.append(tuple(output[:]))

        return outputs

    def evaluator(self, instances_directories):
        """
        Returns:
            heatmap_data (dict): {(time_window, abnormality_parameter_value): best_tuner performance of the
                                  related dataset}
            metric_names (keys): a dictionary keys containing 'loss', 'specificity', 'sensitivity',', accuracy',
                                 and 'geometric_mean'

        """
        results = []
        print('no_instances:', len(instances_directories[self.pattern_name].values()))
        for dataset_directory in instances_directories[self.pattern_name].values():
            result = self.fit_per_dataset(dataset_directory)
            for entry in result:
                results.append(entry)

        output = pd.DataFrame(results, columns=['time_window', 'abnormality_value', 'monitoring_method',
                                                'valid_loss', 'itest_loss', 'btest_loss',
                                                'valid_specificity', 'itest_specificity', 'btest_specificity',
                                                'valid_sensitivity', 'itest_sensitivity', 'btest_sensitivity',
                                                'valid_accuracy', 'itest_accuracy', 'btest_accuracy',
                                                'valid_gmean', 'itest_gmean', 'btest_gmean',
                                                'train_time'])
        output.to_csv(os.path.join(os.getcwd(), 'Early_Stopping', self.pattern_name + '.csv'))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, class_sizes, batch_size, imbalanced_ratio, is_stratified_sampling,
                 is_cost_sensitive, is_equitable, subclass_weights, is_shuffling=True):
        self.dataset = dataset
        self.class_sizes = class_sizes
        self.batch_size = batch_size
        self.imbalanced_ratio = imbalanced_ratio
        self.is_stratified_sampling = is_stratified_sampling
        self.is_cost_sensitive = is_cost_sensitive
        self.is_equitable = is_equitable
        self.subclass_weights = subclass_weights
        self.is_shuffling = is_shuffling

        self.names = list(self.dataset.keys())
        self.no_datapoints = sum(self.class_sizes.values())
        self.batch_size_checker()
        self.true_labels = {name: 0. if name == 'normal' else 1. for name in self.names}

        self.batched_data = self.batch_data_provider()

    def batch_size_checker(self):
        if self.no_datapoints % self.batch_size:
            raise ValueError(f"Please provide a batch size that is dividable by {self.batch_size}")

    def __len__(self):
        return np.ceil(self.no_datapoints / self.batch_size).astype(int)

    def __getitem__(self, batch_id):
        return self.batched_data[batch_id]

    def on_epoch_end(self):
        if self.is_shuffling:
            self.batched_data = self.batch_data_provider()

    def class_sizes_per_batch_specifier(self):
        class_sizes_per_batch = dict()
        no_steps_per_epoch = self.__len__()
        available_sizes = self.class_sizes.copy()
        for batch_id in range(no_steps_per_epoch):
            if self.is_stratified_sampling:
                class_sizes_per_batch[batch_id] = {name: round(self.class_sizes[name] / no_steps_per_epoch)
                                                   for name in self.names}
            else:
                if batch_id == no_steps_per_epoch - 1:
                    sizes = available_sizes
                else:
                    sizes = dict()
                    leftover = self.batch_size
                    for i, name in enumerate(self.names):
                        if name == self.names[-1]:
                            size = leftover
                        else:
                            lower_bound = max(0, leftover - sum([available_sizes[self.names[j]]
                                                                 for j in range(len(self.names)) if j > i]))

                            upper_bound = min(leftover, available_sizes[name])
                            if lower_bound >= upper_bound:
                                size = lower_bound
                            else:
                                size = np.random.randint(lower_bound, upper_bound)

                        sizes[name] = size
                        available_sizes[name] -= size
                        leftover -= size

                    assert leftover == 0

                class_sizes_per_batch[batch_id] = sizes

                assert sum(sizes.values()) == self.batch_size

        return class_sizes_per_batch

    def class_weights_per_batch_specifier(self, class_sizes):
        imbalanced_ratio = class_sizes['normal'] / self.batch_size
        weights = dict()
        for name in self.names:
            if class_sizes[name]:
                if name == 'normal':
                    if self.is_equitable:
                        weights[name] = 1 / imbalanced_ratio
                    else:
                        if not imbalanced_ratio == 1:
                            total_inequity = (1 / (1 - imbalanced_ratio)) * sum([class_sizes[name] *
                                                                                 (1 - self.subclass_weights[name])
                                                                                 for name in self.names
                                                                                 if not name == 'normal' or
                                                                                 not name == 'abnormal_1'])
                            weights[name] = (1 / imbalanced_ratio) - (total_inequity / class_sizes[name])
                        else:
                            weights[name] = 1 / imbalanced_ratio
                else:
                    if self.is_equitable:
                        weights[name] = 1 / (1 - imbalanced_ratio)
                    else:
                        weights[name] = self.subclass_weights[name] / (1 - imbalanced_ratio)
            else:
                weights[name] = 1

        return weights

    def batch_data_provider(self):
        """
        This method creates a list containing the epoch level batch data of (x, y, w) where the weight vector w is
        determine based on two factors: is_stratified and is_equitable.
        1. stratified and equitable:
            batches are created in a way that all of them have similar imbalance ratios as the original
                train dataset; sample weights are defined as 1/imbalance_ratio for normal datapoints, and
                1 / (1 - imbalance_ratio) for abnormal datapoints.

        2. stratified but not equitable:
            batches are created in a way that all of them have similar imbalance ratios as the original
            train dataset; sample weights are defined as 1/imbalance_ratio for normal datapoints, and
            1 / (1 - imbalance_ratio) for abnormal datapoints.

        3. random and equitable
            weights are determined based on the imbalanced ratio of each individual mini-batch data and abnormal
            data points are treated equally in terms of weights

        4. random but not equitable
            weights are determined based on the imbalanced ratio of each individual mini-batch data and abnormal
            data points are receiving different weights.

        Note: the dataset are shuffled after each epoch, and updated the train data placeholder, which is
              batched_data variable.

        Returns:
            epoch_level_batch_data

            batched_data (list): list of batch level datapoints.
                                 Each element of the output is tuple of (x, y, sample_weight).
        """
        batched_data = list()

        class_sizes_per_batch = self.class_sizes_per_batch_specifier()
        start_indices = {name: 0 for name in self.names}

        features = {name: np.reshape(self.dataset[name][0], (self.class_sizes[name], -1, 1))
                    for name in self.names}
        # abnormality_rates = {name: np.reshape(self.dataset[name][1], (self.class_sizes[name], -1))
        #                      for name in self.names}
        for name in self.names:
            shuffler = np.random.permutation(len(features[name]))
            features[name] = features[name][shuffler]

        for batch_id, class_sizes in class_sizes_per_batch.items():
            class_weights = self.class_weights_per_batch_specifier(class_sizes)
            batch_data = dict()
            for name in self.names:
                batch_x = features[name][start_indices[name]: start_indices[name] + class_sizes[name]]
                start_indices[name] += class_sizes[name]
                batch_y = [self.true_labels[name]] * class_sizes[name]
                batch_w = [class_weights[name]] * class_sizes[name]

                batch_data[name] = (batch_x, batch_y, batch_w)

            batch_features = np.concatenate([batch_data[name][0] for name in self.names], axis=0)
            batch_labels = np.concatenate([batch_data[name][1] for name in self.names], axis=0).reshape((-1, 1))
            batch_weights = np.concatenate([batch_data[name][2] for name in self.names], axis=0)

            if not self.is_cost_sensitive:
                batch_weights = None
            batched_data.append((batch_features, batch_labels, batch_weights))

        return batched_data


class MultiObjectiveEarlyStopping(tf.keras.callbacks.Callback):
    """Stop training when the wuc := (relative_val_loss + kapp * relative_val_geometric_mean) is no longer decreasing!

        Args:
            patience (int): number of epochs without improvement in wuc (weight updating criteria)
            kappa (float): the trade-off coefficient between relative loss and gmean improvements
            minimum_no_epochs (int): minimum number of epochs that early stopping starts afterwards.
    """
    def __init__(self, patience, kappas, minimum_no_epochs=1, maximum_no_epochs=1000, verbosity=0):
        super(MultiObjectiveEarlyStopping, self).__init__()
        self.patience = patience
        self.kappas = kappas
        self.minimum_no_epochs = minimum_no_epochs
        self.maximum_no_epochs = maximum_no_epochs
        self.verbosity = verbosity
        # self.best_weights = {'loss': None, 'gmean': None, 'both': None}
        self.best_weights = {'loss': None, 'gmean': None}
        for kappa in kappas:
            self.best_weights[f'both_{kappa}'] = None

    def on_train_begin(self, logs=None):
        start_time = time.time()
        self.start_times = {method_name: start_time for method_name in self.best_weights.keys()}
        self.training_times = dict()
        self.waitings = {method_name: 0 for method_name in self.best_weights.keys()}
        self.best_epochs = {method_name: 0 for method_name in self.best_weights.keys()}
        self.best_losses = {method_name: 100**2 for method_name in self.best_weights.keys()}
        self.best_gmeans = {method_name: 0 for method_name in self.best_weights.keys()}
        self.is_finished = {method_name: False for method_name in self.best_weights.keys()}

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.minimum_no_epochs:
            loss = logs.get("val_loss")
            gmean = logs.get("val_geometric_mean")

            # method loss
            if not self.is_finished['loss']:
                if loss < self.best_losses['loss']:
                    self.best_epochs['loss'] = epoch
                    self.best_losses['loss'] = loss
                    self.best_weights['loss'] = self.model.get_weights()
                    self.waitings['loss'] = 0
                else:
                    self.waitings['loss'] += 1

            # method gmean
            if not self.is_finished['gmean']:
                if gmean > self.best_gmeans['gmean']:
                    self.best_epochs['gmean'] = epoch
                    self.best_gmeans['gmean'] = gmean
                    self.best_weights['gmean'] = self.model.get_weights()
                    self.waitings['gmean'] = 0
                else:
                    self.waitings['gmean'] += 1

            # method bi-objective
            for kappa in self.kappas:
                loss_percentage_change = (loss - self.best_losses[f'both_{kappa}']) / \
                                         (self.best_losses[f'both_{kappa}'] + 1e-20)

                gmean_percentage_change = (gmean - self.best_gmeans[f'both_{kappa}']) / \
                                          (self.best_gmeans[f'both_{kappa}'] + 1e-20)
                weight_updating_criterion = loss_percentage_change - kappa * gmean_percentage_change

                if not self.is_finished[f'both_{kappa}']:
                    if weight_updating_criterion < 0:
                        self.best_epochs[f'both_{kappa}'] = epoch
                        self.best_losses[f'both_{kappa}'] = loss
                        self.best_gmeans[f'both_{kappa}'] = gmean
                        self.best_weights[f'both_{kappa}'] = self.model.get_weights()
                        self.waitings[f'both_{kappa}'] = 0
                    else:
                        self.waitings[f'both_{kappa}'] += 1

            if not self.is_finished['loss']:
                if self.waitings['loss'] > self.patience or epoch >= self.maximum_no_epochs - 1:
                    self.is_finished['loss'] = True
                    self.training_times['loss'] = time.time() - self.start_times['loss']

            if not self.is_finished['gmean']:
                if self.waitings['gmean'] > self.patience or epoch >= self.maximum_no_epochs - 1:
                    self.is_finished['gmean'] = True
                    self.training_times['gmean'] = time.time() - self.start_times['gmean']

            for kappa in self.kappas:
                if not self.is_finished[f'both_{kappa}']:
                    if self.waitings[f'both_{kappa}'] > self.patience or epoch >= self.maximum_no_epochs - 1:
                        self.is_finished[f'both_{kappa}'] = True
                        self.training_times[f'both_{kappa}'] = time.time() - self.start_times[f'both_{kappa}']

            if all(self.is_finished.values()):
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.verbosity:
            print(f"\n The best epochs are : {self.best_epochs}")


class Experimenter:

    @staticmethod
    def instace_directory_finder(pattern_name, instances):
        ms_datasets_path = os.path.join(os.getcwd(), 'DS_for_MS', pattern_name + '_*.txt')
        dataset_directories = glob.glob(ms_datasets_path)
        remaining_datasets_path = os.path.join(os.getcwd(), 'Remaining_DS', pattern_name + '_*.txt')
        remaining_files = glob.glob(remaining_datasets_path)
        dataset_directories.extend(remaining_files)

        assert len(dataset_directories) == 250
        instances_dictionaries = dict()
        for dataset_directory in dataset_directories:
            with open(file=dataset_directory, mode='r') as readfile:
                file = json.load(readfile)

            time_window = file["time_window"]
            abnormality_value = file['abnormality_parameters']['parameter_value']
            if (time_window, abnormality_value) in zip(instances['time_windows'], instances['abnormality_values']):
                instances_dictionaries[(time_window, abnormality_value)] = dataset_directory

        return instances_dictionaries

    def experiment_conductor(self, imbalanced_ratio, pattern_name, subclass_weights, minimum_no_epochs,
                             maximum_no_epochs, patience, kappas, instances, method_name="BayesianOptimization"):

        strategy_directory = os.path.join(os.getcwd(),
                                          "Offline_Best_Models_Performance",
                                          "%.2f" % imbalanced_ratio + "_imbalanced_ratio",
                                          method_name,
                                          pattern_name + ".txt")
        with open(file=strategy_directory, mode='r') as readfile:
            strategies_info = json.load(readfile)
        best_strategy = strategies_info["The_best_strategy"]

        best_hp_directory = os.path.join(os.getcwd(),
                                         "Offline_Best_Models",
                                         "%.2f" % imbalanced_ratio + "_imbalanced_ratio", method_name,
                                         pattern_name + ".txt")
        with open(file=best_hp_directory, mode='r') as readfile:
            best_models = json.load(readfile)
        best_model = best_models[best_strategy]

        classifiers = Classifiers(hp=best_model, imbalanced_ratio=imbalanced_ratio, best_strategy=best_strategy,
                                  pattern_name=pattern_name, subclass_weights=subclass_weights,
                                  minimum_no_epochs=minimum_no_epochs, maximum_no_epochs=maximum_no_epochs,
                                  patience=patience, kappas=kappas)

        classifiers.evaluator(instances)

    def main(self):
        with open(file=os.path.join(os.getcwd(), "Input_data", "instances_info.txt"), mode='r') as readfile:
            instances_info = json.load(readfile)

        with open(file=os.path.join(os.getcwd(), "Input_data",
                                    "offline_classification_parameters.txt"), mode='r') as readfile:
            classification_parameters = json.load(readfile)

        with open(file=os.path.join(os.getcwd(), "Input_data",
                                    "early_stopping_instances.txt"), mode='r') as readfile:
            early_stopping_instances = json.load(readfile)

        pattern_names = instances_info['pattern_names']

        selected_instances_directories = dict()
        for pattern_name in pattern_names:
            selected_instances_directories[pattern_name] = \
                self.instace_directory_finder(pattern_name, early_stopping_instances[pattern_name])

        # kappas = np.arange(0.0, 30, 0.4)
        # kappas = np.geomspace(1e-2, 50, 15)
        kappas = np.arange(0.1, 10, 0.1).tolist() + np.arange(11, 20, 1).tolist() + [25, 30, 40, 50, 100]

        data_frames = []

        for pattern_name in pattern_names:
            self.experiment_conductor(pattern_name=pattern_name,
                                      imbalanced_ratio=instances_info["imbalanced_ratio"],
                                      subclass_weights=classification_parameters["subclass_weights"],
                                      minimum_no_epochs=classification_parameters["minimum_no_epochs"],
                                      maximum_no_epochs=classification_parameters["maximum_no_epochs"],
                                      kappas=kappas,
                                      patience=classification_parameters["patience"],
                                      instances=selected_instances_directories)

            file_name = os.path.join(os.getcwd(), 'Early_Stopping', f'{pattern_name}.csv')
            df = pd.read_csv(file_name)
            df = df.rename(columns={'Unnamed: 0': 'Pattern-Name'})
            df['Pattern-Name'] = pattern_name
            data_frames.append(df)

        concatenated_df = pd.concat(data_frames)

        concatenated_df.to_csv(os.path.join(os.getcwd(), "Early_Stopping", 'early_stoppings.csv'), index=False)


if __name__ == "__main__":
    experimenter = Experimenter()
    experimenter.main()
