import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa: E402

import re

import json

from itertools import product

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.optimizers import Nadam

from tensorflow.keras import Model, Input

from tensorflow.keras.layers import LSTM, Dense, Bidirectional, RNN

from tensorflow_addons.rnn import LayerNormLSTMCell

import pandas as pd


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


class BinaryControlChartPatterns:
    """
    This class generates a set of simulated datasets for the problem of control chart pattern recognition based on the
    equations in the paper of:
        A weighted support vector machine method for control chart pattern recognition, written by Petros
        Xanthopoulos and Talayeh Razzaghi.

    """

    def __init__(self, name, dataset_distribution, time_window=10, imbalanced_ratio=0.95,
                 abnormality_parameters=None, visualization=False, chart_type="mixed"):
        """
        Args:
            name (str): name of the generated dataset
            dataset_distribution (dict): a dictionary determining the number of data points in each set and subclass

                {train: {normal: 570, abnormal_0.2: 6, abnormal_0.4: 6, ..., abnormal_0.8: 6, abnormal_1: 6}
                valid: {normal: 190, abnormal_0.2: 2, abnormal_0.4: 2, ..., abnormal_0.8: 2, abnormal_1: 2}
                imbalanced_test: {normal: 190, abnormal_0.2: 2, abnormal_0.4: 2, ..., abnormal_0.8: 2, abnormal_1: 2}
                balanced_test: {normal: 100, abnormal_0.2: 20, abnormal_0.4: 20, ..., abnormal_0.8: 20, abnormal_1: 20}}

            time_window (int): the length of the generated time-series.
            imbalanced_ratio (float): a rate in which the normal and abnormal data are distributed
            abnormality_parameters (dict): dictionary containing information about the type of abnormality and
                                           corresponding abnormality parameter
            visualization (bool): should we visualize the generated data set

        """
        if abnormality_parameters is None:
            abnormality_parameters = {'pattern_name': 'uptrend', 'parameter_value': 0.05}
        self.name = name
        self.dataset_distribution = dataset_distribution
        self.time_window = time_window
        self.imbalanced_ratio = imbalanced_ratio
        self.abnormality_parameters = abnormality_parameters

        if chart_type == 'mixed':
            self.abnormality_rates = [1, 0.8, 0.6, 0.4, 0.2, 0]
        else:
            self.abnormality_rates = [1, 0]

        self.raw_data = self.data_generator()

        if visualization:
            self.control_chart_pattern_visualizer(self.raw_data['train'])

    def raw_data_initializer(self):
        """This method creates a random normal term of the supposed generating raw_data."""
        distribution = self.dataset_distribution
        no_samples = sum([sum(distribution[set_name].values()) for set_name in distribution.keys()])
        normal_term = np.random.normal(size=(no_samples, self.time_window))
        np.random.shuffle(normal_term)

        initial_data = dict()
        start_index = 0
        for set_name in distribution.keys():
            initial_data[set_name] = dict()
            for class_name in distribution[set_name].keys():
                initial_data[set_name][class_name] = normal_term[start_index:
                                                                 start_index + distribution[set_name][class_name]]
                start_index += distribution[set_name][class_name]

        return initial_data

    def data_generator(self):
        """
        This method appends abnormality term to the generated initial normal parts based on the equations:
            trend: a(t) = eps(t)  + trend_slope * t,
            shift: a(t) = eps(t) + shift_value,
            cyclic: a(t) = eps(t) + amplitude * SIN(2*PI/period)
            systematic: a(t) = eps(t) + magnitude * (-1)**t
            stratification: a(t) = eps'(t),
        where eps(t) eps'(t) has the Standard Normal distribution, i.e. Z(0, 1), and normal distribution with mean
        equal to zero and standard deviation equal to "stratification_value".

        Args:
            initial_data (dict): dictionary of initial normal terms.

        Returns:
            the dictionary of 'raw_data' with independent segments of data sets with the following structure:
                {
                train: {normal: (x, y=0), "abnormal_0.2": (x, y=0.2), ..., "abnormal_1": (x, y=1)},
                valid: {normal: (x, y=0), "abnormal_0.2": (x, y=0.2), ..., "abnormal_1": (x, y=1)},
                imbalanced_test: {normal: (x, y=0), "abnormal_0.2": (x, y=0.2), ..., "abnormal_1": (x, y=1)},
                balanced_test: {normal: (x, y=0), "abnormal_0.2": (x, y=0.2), ..., "abnormal_1": (x, y=1)}
                }

        """

        def abnormality_appender(set_name, class_name, abnormality_rate):
            """This function modifies "raw_data" with respect to the abnormality_rate for the given "class_name" class
            inside the "set_name" set."""
            pattern_name = self.abnormality_parameters["pattern_name"]
            parameter_value = self.abnormality_parameters["parameter_value"]
            time_window = self.time_window
            normal_term = initial_data[set_name][class_name]

            no_normal_steps = round((1 - abnormality_rate) * time_window)
            no_abnormal_steps = time_window - no_normal_steps

            normal_steps = np.zeros(no_normal_steps)
            if pattern_name == 'uptrend':
                abnormal_steps = np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
            elif pattern_name == 'downtrend':
                abnormal_steps = - np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
            elif pattern_name == 'upshift':
                abnormal_steps = np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
            elif pattern_name == 'downshift':
                abnormal_steps = - np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
            elif pattern_name == "systematic":
                abnormal_steps = np.array(list(map(lambda t: parameter_value if t % 2 == 0 else -1 * parameter_value,
                                                   range(no_abnormal_steps))))
            elif pattern_name == "stratification":
                abnormal_steps = parameter_value * np.random.normal(size=no_abnormal_steps)
            elif pattern_name == "cyclic":
                period = 8
                abnormal_steps = np.array(list(map(lambda t: parameter_value * np.sin(2 * np.pi * t / period),
                                                   range(no_abnormal_steps))))
            else:
                raise ValueError("pattern name NOT found!")

            trend_term = np.concatenate((normal_steps, abnormal_steps))

            features = normal_term + trend_term
            labels = abnormality_rate * np.ones(features.shape[0])

            return features, labels

        initial_data = self.raw_data_initializer()
        raw_data = dict()
        for set_name in initial_data.keys():
            raw_data[set_name] = dict()
            for class_name in initial_data[set_name].keys():
                abnormality_rate = self.abnormality_rate_sepcifier(class_name)
                raw_data[set_name][class_name] = abnormality_appender(set_name, class_name, abnormality_rate)

        return raw_data

    def data_provider(self):
        raw_data = self.data_generator()
        normalized_data, scaler = self.data_preprocessor(raw_data)
        return normalized_data, scaler

    def abnormality_rate_sepcifier(self, subclass_name):
        """This method determines the abnormality rate based on a given subclass name, e.g.:
            if subclass_name = 'abnormal_0.6', then abnormality_rate = 0.6
        """
        if subclass_name == 'normal':
            abnormality_rate = 0
        else:
            abnormality_rate = float(re.findall(r"[-+]?\d*\.\d+|\d+", subclass_name)[0])

        assert abnormality_rate in self.abnormality_rates

        return abnormality_rate

    def control_chart_pattern_visualizer(self, dataset):
        """This function visualizes and saves the generated time-series data

        Args:
            dataset (dict): train dictionary containing normal and abnormal time-series and labels
                {normal: (x, y=0), abnormal_0.2: (x,y=0.2), ..., abnormal_1: (x, y=1)}

        Returns:
            None

        """
        figure, axes = plt.subplots(2, 3, figsize=(12, 6))
        figure.tight_layout()

        plot_id = 0
        for class_name in dataset.keys():
            timeseries, labels = dataset[class_name]
            eta = self.abnormality_rate_sepcifier(class_name)
            index = int(np.random.choice(range(len(timeseries)), 1, replace=False))
            ax = axes[divmod(plot_id, 3)]
            ax.plot(timeseries[index], color="#000032", linewidth=2.2, label='Normal')
            if eta > 0:
                time_window = self.time_window
                abnormality_start_point = round(time_window * (1 - eta))
                ax.plot(list(range(abnormality_start_point, time_window)),
                        timeseries[index][abnormality_start_point:],
                        color='#5c0000', linewidth=2.4, label="Abnormal")

            plot_id += 1

            figure.savefig(f'{self.name}.png', dpi=300)

    def test_data_generator(self, pattern_name, time_window, parameter_value, no_tests, detection_horizon):
        normal_term = np.random.normal(size=(no_tests, time_window + detection_horizon))
        np.random.shuffle(normal_term)
        no_normal_steps = time_window - 1
        no_abnormal_steps = detection_horizon + 1

        normal_steps = np.zeros(no_normal_steps)
        if pattern_name == 'uptrend':
            abnormal_steps = np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
        elif pattern_name == 'downtrend':
            abnormal_steps = - np.array(list(map(lambda t: parameter_value * t, range(no_abnormal_steps))))
        elif pattern_name == 'upshift':
            abnormal_steps = np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
        elif pattern_name == 'downshift':
            abnormal_steps = - np.array(list(map(lambda t: parameter_value, range(no_abnormal_steps))))
        elif pattern_name == "systematic":
            abnormal_steps = np.array(list(map(lambda t: parameter_value if t % 2 == 0 else -1 * parameter_value,
                                               range(no_abnormal_steps))))
        elif pattern_name == "stratification":
            abnormal_steps = parameter_value * np.random.normal(size=no_abnormal_steps)
        elif pattern_name == "cyclic":
            period = 8
            abnormal_steps = np.array(list(map(lambda t: parameter_value * np.sin(2 * np.pi * t / period),
                                               range(no_abnormal_steps))))
        else:
            raise ValueError("pattern name NOT found!")

        trend_term = np.concatenate((normal_steps, abnormal_steps))

        features = normal_term + trend_term
        # labels = np.ones(features.shape[0])

        # return features, labels
        return features

    def data_preprocessor(self, raw_data):
        """
        This method splits the 'raw_data' data points inside the file_name into train, valid, test sets
        according to data_distribution parameter, and then, normalizes the timeseries data and transforms zero-one
        labels into the one-hot encoded vectors. Finally, it writes the processed dataset back into the file_name
        text file with "normalized_dataset" key. This dataset, then, is used both in the feature extraction
        and classification processes.

        Args:
            file_name (str): a file_name wherein the 'raw_data' dataset is stored.

        Returns:
            None
        """
        timeseries = np.concatenate(([raw_data['train'][class_name][0] for class_name in raw_data['train'].keys()]))

        scaler = StandardScaler()
        scaler.fit(timeseries)

        normalized_dataset = dict()

        for set_name in raw_data.keys():
            normalized_dataset[set_name] = dict()
            for class_name in raw_data[set_name].keys():
                normalized_features = scaler.transform(raw_data[set_name][class_name][0])
                labels = raw_data[set_name][class_name][1]
                normalized_dataset[set_name][class_name] = normalized_features, labels

                assert np.all(np.isfinite(normalized_features))
                assert np.all(np.isfinite(labels))
                assert not np.any(np.isnan(normalized_features))
                assert not np.any(np.isnan(labels))
                assert not np.any(np.isinf(normalized_features))
                assert not np.any(np.isinf(labels))

        return normalized_dataset, scaler


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

    def __init__(self, hp, strategy_name, imbalanced_ratio, pattern_name, minimum_no_epochs,
                 maximum_no_epochs, patience, is_cost_sensitive, chart_type):
        """
        Args:
            hp (dict): best hyperparameters.
        """
        self.hp = hp

        self.is_stratified_sampling = "Stratified" in strategy_name
        # self.is_cost_sensitive = "cost-sensitive" in strategy_name
        self.is_cost_sensitive = is_cost_sensitive
        if self.is_cost_sensitive:
            self.is_equitable = True
        else:
            self.is_equitable = None

        self.imbalanced_ratio = imbalanced_ratio
        self.pattern_name = pattern_name

        self.minimum_no_epochs = minimum_no_epochs
        self.maximum_no_epochs = maximum_no_epochs
        self.patience = patience

        self.kappa = hp["kappa"]

        # self.subclass_weights = self.subclass_weight_specifier(hp["alpha"], hp["is_emphasizing"])
        if chart_type == 'mixed':
            self.subclass_weights = {'normal': 1, "abnormal_0.2": 1, "abnormal_0.4": 1,
                                     "abnormal_0.6": 1, "abnormal_0.8": 1, "abnormal_1": 1}
        else:
            self.subclass_weights = {'normal': 1, "abnormal_1": 1}

        self.metric_names = ['loss', 'specificity', 'sensitivity', 'accuracy', 'geometric_mean']

    # @staticmethod
    # def subclass_weight_specifier(alpha, is_emphasizing):
    #     """this method determines the weight retaining parameters of subclasses using power function
    #     eta ** alpha, where eta shows the percentage of abnormal signals for a subclass."""
    #     subclass_weights = dict()
    #     for eta in [0.2, 0.4, 0.6, 0.8, 1]:
    #         if is_emphasizing:
    #             subclass_weights["abnormal_" + str(eta)] = 1 + eta ** alpha
    #         else:
    #             subclass_weights["abnormal_" + str(eta)] = 1 + (1 - eta) ** alpha
    #
    #     subclass_weights["normal"] = max(subclass_weights.values())
    #
    #     return subclass_weights

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

    def valid_data_provider(self, datapoints, datapoints_sizes):
        names = list(datapoints.keys())
        features = {name: np.reshape(datapoints[name][0], (datapoints_sizes[name], -1, 1))
                    for name in names}
        labels = {name: np.reshape(np.ceil(datapoints[name][1]), (datapoints_sizes[name], -1))
                  for name in names}

        weights = {}

        for name in names:
            if name == "normal":
                if self.is_equitable:
                    weights[name] = np.array([1 / self.imbalanced_ratio] * datapoints_sizes[name])
                else:
                    # total_inequity = (1 / (1 - self.imbalanced_ratio)) * \
                    #                  sum([datapoints_sizes[name] * (1 - self.subclass_weights[name])
                    #                       for name in names if name != 'normal'])
                    # weights[name] = np.array([1 / self.imbalanced_ratio -
                    #                           total_inequity / datapoints_sizes[name]] * datapoints_sizes[name])
                    upperbound = max(self.subclass_weights.values())
                    total_inequity = (1 / (1 - self.imbalanced_ratio)) * sum(
                        [datapoints_sizes[name] * (upperbound - self.subclass_weights[name])
                         for name in names if not name == 'normal'])

                    weights[name] = np.array([(self.subclass_weights[name] / self.imbalanced_ratio) - \
                                              (total_inequity / datapoints_sizes[name])] * datapoints_sizes[name])
            else:
                if self.is_equitable:
                    weights[name] = np.array([1 / (1 - self.imbalanced_ratio)] * datapoints_sizes[name])
                else:
                    weights[name] = np.array([self.subclass_weights[name] / (1 - self.imbalanced_ratio)] *
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

    # def learning_curve_visualizer(self, history, time_param):
    #
    #     fig, axes = plt.subplots(2, 1, sharex=True, squeeze=False)
    #
    #     axes[(0, 0)].plot(history.history['loss'], color='k', label='train loss', linewidth=1.5)
    #     axes[(0, 0)].plot(history.history['val_loss'], color='b', label='valid loss')
    #     axes[(0, 0)].legend(loc='upper center', shadow=True, fontsize='x-small')
    #     axes[(0, 0)].set(ylabel='loss', title='learning curve')
    #     axes[(0, 0)].grid()
    #
    #     axes[(1, 0)].plot(history.history['val_geometric_mean'], color='r', label='valid G-mean')
    #     axes[(1, 0)].legend(loc='lower right', shadow=True, fontsize='x-small')
    #     axes[(1, 0)].grid()
    #     axes[(1, 0)].set(xlabel='epochs', ylabel='G-mean')
    #     fig.tight_layout()
    #
    #     strategy_fname = f"Strf={str(self.is_stratified_sampling)[0]}-" \
    #                      f"CS={str(self.is_cost_sensitive)[0]}-" \
    #                      f"EQ={str(self.is_stratified_sampling)[0]}"
    #
    #     curves_paths = os.path.join(os.getcwd(), 'Offline_Learning_Curves', 'In_heatmaps', self.pattern_name,
    #                                 strategy_fname)
    #     curves_save_fname = os.path.join(curves_paths, time_param + '.png')
    #
    #     os.makedirs(curves_paths, exist_ok=True)
    #
    #     plt.savefig(curves_save_fname)
    #
    #     plt.close()

    def fit_per_dataset(self, dataset, time_window, abnormality_value, dataset_distribution):

        train_datapoints = dataset['train']

        valid_data, valid_batch_size = self.valid_data_provider(dataset['valid'], dataset_distribution['valid'])

        batch_size = self.hp["batch_size"]

        gmean_callback = MultiObjectiveEarlyStopping(patience=self.patience,
                                                     kappa=self.kappa,
                                                     minimum_no_epochs=self.minimum_no_epochs,
                                                     verbosity=0)

        time_param = f"T={time_window}-prm={abnormality_value}"
        strategy_fname = f"Strf={str(self.is_stratified_sampling)[0]}-" \
                         f"CS={str(self.is_cost_sensitive)[0]}-" \
                         f"EQ={str(self.is_stratified_sampling)[0]}"

        # history_callback_dir = os.path.join(os.getcwd(), 'Offline_Histories', 'In_heatmaps', self.pattern_name,
        #                                     strategy_fname)
        # os.makedirs(history_callback_dir, exist_ok=True)
        # history_fname = os.path.join(history_callback_dir, time_param + '.csv')
        # history_callback = tf.keras.callbacks.CSVLogger(history_fname, separator=",", append=False)

        assert time_window == len(train_datapoints['normal'][0][0])
        model = self.build(time_window)

        # # check the parameters
        # print("is_stratified_sampling", self.is_stratified_sampling)
        # print("is_cost_sensitive", self.is_cost_sensitive)
        # print("is_equitable", self.is_equitable)
        # print("subclass_weights", self.subclass_weights)

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
                            # callbacks=[gmean_callback, history_callback],
                            callbacks=[gmean_callback],
                            validation_data=valid_data,
                            validation_batch_size=valid_batch_size,
                            verbose=0)

        trained_classifier = history.model

        return trained_classifier

    def detection_time_quantifier(self):
        pass

    #
    # def heatmap_calculator(self):
    #     """
    #     Returns:
    #         heatmap_data (dict): {(time_window, abnormality_parameter_value): best_tuner performance of the
    #                               related dataset}
    #         metric_names (keys): a dictionary keys containing 'loss', 'specificity', 'sensitivity',', accuracy',
    #                              and 'geometric_mean'
    #
    #     """
    #     imbalanced_heatmap_data = {}
    #     balanced_heatmap_data = {}
    #
    #     ms_datasets_path = os.path.join(os.getcwd(), 'DS_for_MS', self.pattern_name + '_*.txt')
    #     dataset_directories = glob.glob(ms_datasets_path)
    #     remaining_datasets_path = os.path.join(os.getcwd(), 'Remaining_DS', self.pattern_name + '_*.txt')
    #     remaining_files = glob.glob(remaining_datasets_path)
    #     dataset_directories.extend(remaining_files)
    #
    #     assert len(dataset_directories) == 250
    #
    #     # # todo: remove this part! for test only!
    #     results = []
    #     for dataset_directory in dataset_directories:
    #         result = self.fit_per_dataset(dataset_directory)
    #         results.append(result)
    #
    #     # with ProcessPoolExecutor(max_workers=1) as executor:  # todo:
    #     #
    #     #     futures = [executor.submit(self.fit_per_dataset, dataset_directory)
    #     #                for dataset_directory in dataset_directories]  # todo:
    #     #     results = [future.result() for future in futures]
    #
    #     for result in results:
    #         imbalanced_heatmap_data[(result[0], result[1])] = dict(zip(self.metric_names, np.array(result[2])))
    #         balanced_heatmap_data[(result[0], result[1])] = dict(zip(self.metric_names, np.array(result[3])))
    #
    #     return imbalanced_heatmap_data, balanced_heatmap_data
    #
    # def heatmap_visulizer(self, data, is_balanced):
    #     """
    #     This method visualizes the heatmaps.
    #     Args:
    #         data (dict): {(time_window, abnormality_parameter_value): best_tuner performance of the
    #                               related dataset}
    #     Returns:
    #         None
    #
    #     """
    #     df = pd.DataFrame.from_dict(data, orient='index')
    #     reindexed_df = df.reset_index()
    #     renamed_df = reindexed_df.rename(columns={'level_0': 'Time window', 'level_1': 'Abnormality parameter'})
    #
    #     for metric_name in self.metric_names:
    #         if not (metric_name == "loss"):
    #             lower_range = 0
    #             upper_range = 1
    #         else:
    #             lower_range = None
    #             upper_range = None
    #
    #         heatmap_data = renamed_df.pivot_table(values=metric_name, index='Time window',
    #                                               columns='Abnormality parameter', aggfunc=np.sum)
    #         heatmap_data = heatmap_data.sort_index(ascending=False)
    #         heatmap_data = heatmap_data.sort_index(1)
    #
    #         plt.figure()
    #         # sb.heatmap(heatmap_data, linewidths=1, annot=False, cmap='coolwarm', vmin=lower_range, vmax=upper_range)
    #         sb.heatmap(heatmap_data, linewidths=0, annot=False, cmap='YlGnBu', vmin=lower_range, vmax=upper_range)
    #
    #         strategy_fname = f"Strf={str(self.is_stratified_sampling)[0]}-" \
    #                          f"CS={str(self.is_cost_sensitive)[0]}-" \
    #                          f"EQ={str(self.is_stratified_sampling)[0]}"
    #         heatmap_directory = os.path.join(os.getcwd(), 'Offline_Heatmaps', strategy_fname)
    #         os.makedirs(heatmap_directory, exist_ok=True)
    #
    #         heatmap_results_directory = os.path.join(os.getcwd(), 'Offline_Heatmaps_Results', strategy_fname)
    #         os.makedirs(heatmap_results_directory, exist_ok=True)
    #
    #         if is_balanced:
    #             fig_name = 'balanced_' + self.pattern_name + '_' + metric_name + '.png'
    #             csv_fname = 'balanced_' + self.pattern_name + '_' + metric_name + '.csv'
    #         else:
    #             fig_name = 'imbalanced_' + self.pattern_name + '_' + metric_name + '.png'
    #             csv_fname = 'imbalanced_' + self.pattern_name + '_' + metric_name + '.csv'
    #
    #         plt.savefig(os.path.join(heatmap_directory, fig_name))
    #
    #         saving_fname = os.path.join(heatmap_results_directory, csv_fname)
    #         pd.DataFrame.to_csv(heatmap_data, saving_fname, header=True, index=True)
    #         plt.close()


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

    def __init__(self, patience=0, kappa=1.1, minimum_no_epochs=1, verbosity=0):
        super(MultiObjectiveEarlyStopping, self).__init__()
        self.patience = patience
        self.kappa = kappa
        self.minimum_no_epochs = minimum_no_epochs
        self.verbosity = verbosity
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_epoch = 0
        self.best_loss = 10 ** 10
        self.best_gmean = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.minimum_no_epochs:
            loss = logs.get("val_loss")
            gmean = logs.get("val_geometric_mean")

            loss_percentage_change = (loss - self.best_loss) / (self.best_loss + 1e-20)
            gmean_percentage_change = (gmean - self.best_gmean) / (self.best_gmean + 1e-20)
            weight_updating_criterion = loss_percentage_change - self.kappa * gmean_percentage_change

            if weight_updating_criterion < 0:
                self.best_epoch = epoch
                self.best_loss = loss
                self.best_gmean = gmean
                self.best_weights = self.model.get_weights()
                self.wait = 0
            else:
                self.wait += 1

            if self.wait > self.patience:
                self.model.stop_training = True
                if self.verbosity:
                    print(f"\n Early stopping happened at epoch {epoch}!")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.best_epoch > 0:
            if self.verbosity:
                print(f"\n The best weights for the network are retrieved from epoch {self.best_epoch}")


class Experimenter:
    def __init__(self, pattern_name, time_window, abnormality_value, imbalanced_ratio=0.95, detection_horizon=120,
                 sizes=None):

        if sizes is None:
            sizes = (3000, 1000, 300)

        self.no_trains = sizes[0]
        self.no_valids = sizes[1]
        self.no_tests = sizes[2]

        self.imbalanced_ratio = imbalanced_ratio
        self.pattern_name = pattern_name
        self.time_window = time_window
        self.abnormality_value = abnormality_value

        self.detection_horizon = detection_horizon

        self.mixed_distribution = None
        self.convn_distribution = None
        self.datasets_sizes_specifier()

        self.mixed_data = None
        self.mixed_scaler = None
        self.convn_data = None
        self.convn_scaler = None
        self.test_data = None

        self.datasets_initializer()

    def datasets_sizes_specifier(self):
        no_normals_train = round(self.no_trains * self.imbalanced_ratio)
        no_abnormals_train = round(self.no_trains * (1 - self.imbalanced_ratio))
        no_normals_valid = round(self.no_valids * self.imbalanced_ratio)
        no_abnormals_valid = round(self.no_valids * (1 - self.imbalanced_ratio))
        no_subabnormals_train = round(no_abnormals_train / 5)
        no_subabnormals_valid = round(no_abnormals_valid / 5)
        assert no_normals_train + no_abnormals_train == self.no_trains

        mixed_distribution = {"train": {"normal": no_normals_train,
                                        "abnormal_0.2": no_subabnormals_train,
                                        "abnormal_0.4": no_subabnormals_train,
                                        "abnormal_0.6": no_subabnormals_train,
                                        "abnormal_0.8": no_subabnormals_train,
                                        "abnormal_1": no_subabnormals_train},
                              "valid": {"normal": no_normals_valid,
                                        "abnormal_0.2": no_subabnormals_valid,
                                        "abnormal_0.4": no_subabnormals_valid,
                                        "abnormal_0.6": no_subabnormals_valid,
                                        "abnormal_0.8": no_subabnormals_valid,
                                        "abnormal_1": no_subabnormals_valid}}

        convn_distribution = {"train": {"normal": no_normals_train, "abnormal_1": no_abnormals_train},
                              "valid": {"normal": no_normals_valid, "abnormal_1": no_abnormals_valid}}

        self.mixed_distribution = mixed_distribution
        self.convn_distribution = convn_distribution

    def datasets_initializer(self):
        mixed_charts = BinaryControlChartPatterns(name='mixed_data',
                                                  dataset_distribution=self.mixed_distribution,
                                                  time_window=self.time_window,
                                                  imbalanced_ratio=self.imbalanced_ratio,
                                                  abnormality_parameters={'pattern_name': self.pattern_name,
                                                                          'parameter_value': self.abnormality_value},
                                                  chart_type='mixed')
        mixed_data, mixed_scaler = mixed_charts.data_provider()

        convn_charts = BinaryControlChartPatterns(name='convn_data',
                                                  dataset_distribution=self.convn_distribution,
                                                  time_window=self.time_window,
                                                  imbalanced_ratio=self.imbalanced_ratio,
                                                  abnormality_parameters={'pattern_name': self.pattern_name,
                                                                          'parameter_value': self.abnormality_value},
                                                  chart_type='convn')
        convn_data, convn_scaler = convn_charts.data_provider()

        self.mixed_data = mixed_data
        self.convn_data = convn_data

        self.mixed_scaler = mixed_scaler
        self.convn_scaler = convn_scaler

        test_data = mixed_charts.test_data_generator(self.pattern_name, self.time_window, self.abnormality_value,
                                                     self.no_tests, self.detection_horizon)

        self.test_data = test_data

    def trainer(self):
        # load training info
        with open(file=os.path.join(os.getcwd(), "Input_data",
                                    "offline_classification_parameters.txt"), mode='r') as readfile:
            classification_parameters = json.load(readfile)

        # load best model
        strategy_directory = os.path.join(os.getcwd(), "Offline_Best_Models_Performance", "0.95_imbalanced_ratio",
                                          "BayesianOptimization", self.pattern_name + ".txt")
        with open(file=strategy_directory, mode='r') as readfile:
            strategies_info = json.load(readfile)
        best_strategy = strategies_info["The_best_strategy"]

        best_hp_directory = os.path.join(os.getcwd(), "Offline_Best_Models", "0.95_imbalanced_ratio",
                                         "BayesianOptimization", self.pattern_name + ".txt")
        with open(file=best_hp_directory, mode='r') as readfile:
            best_models = json.load(readfile)
        best_model = best_models[best_strategy]

        # train classifiers
        trained_classifiers = {}
        for chart_type in ['mixed', 'convn']:
            for is_cost_sensitive in [True, False]:
                classifier = Classifiers(hp=best_model,
                                         imbalanced_ratio=self.imbalanced_ratio,
                                         strategy_name=best_strategy,
                                         pattern_name=self.pattern_name,
                                         minimum_no_epochs=classification_parameters["minimum_no_epochs"],
                                         maximum_no_epochs=classification_parameters["maximum_no_epochs"],
                                         patience=classification_parameters["patience"],
                                         is_cost_sensitive=is_cost_sensitive,
                                         chart_type=chart_type)

                if chart_type == 'mixed':
                    trained_classifier = classifier.fit_per_dataset(self.mixed_data, self.time_window,
                                                                    self.abnormality_value, self.mixed_distribution)
                else:
                    trained_classifier = classifier.fit_per_dataset(self.convn_data, self.time_window,
                                                                    self.abnormality_value, self.convn_distribution)

                trained_classifiers[f'{chart_type}_CS={str(is_cost_sensitive)[0]}'] = trained_classifier

        return trained_classifiers

    # def detection_time_evaluator(self, classifier, scaler):
    #
    #     windowed_data = self.rolling_window_strider(self.test_data, self.time_window)
    #
    #     no_rolls = self.test_data.shape[-1] - self.time_window + 1
    #
    #     reshaped_windows = windowed_data.reshape(-1, self.time_window)
    #
    #     normalized_windows = scaler.transform(reshaped_windows)
    #
    #     normalized_windows = normalized_windows.reshape(self.no_tests, no_rolls, self.time_window)
    #
    #     normalized_windows = normalized_windows.reshape(-1, self.time_window, 1)
    #
    #     predictions = classifier.predict(normalized_windows) > 0.5
    #
    #     reshaped_predictions = predictions.reshape(self.no_tests, no_rolls)
    #
    #     detection_times = np.argmax(reshaped_predictions, axis=1) + 1
    #
    #     mask = np.all(~reshaped_predictions, axis=1)
    #     detection_times = detection_times.astype(np.float64)
    #     detection_times[mask] = np.nan
    #
    #     return detection_times

    def detection_time_evaluator(self, classifier, scaler):

        windowed_data = self.rolling_window_strider(self.test_data, self.time_window)

        no_rolls = self.test_data.shape[-1] - self.time_window + 1

        reshaped_windows = windowed_data.reshape(-1, self.time_window)

        normalized_windows = scaler.transform(reshaped_windows)

        normalized_windows = normalized_windows.reshape(self.no_tests, no_rolls, self.time_window)

        normalized_windows = normalized_windows.reshape(-1, self.time_window, 1)

        predictions = classifier.predict(normalized_windows) > 0.5

        reshaped_predictions = predictions.reshape(self.no_tests, no_rolls)

        detection_times = np.argmax(reshaped_predictions, axis=1) + 1

        mask = np.all(~reshaped_predictions, axis=1)
        detection_times = detection_times.astype(np.float64)
        detection_times[mask] = np.nan

        # calculationn of undetected samples percentage and TASRID, and TAR
        undetected_samples_percentage = np.isnan(detection_times).sum() / self.test_data.shape[0]
        true_alert_streaks_rates_from_initial_detection = np.apply_along_axis(
            self.calculate_longest_consecutive_true_detection,
            axis=1,
            arr=reshaped_predictions) / no_rolls

        mean_tasrid = np.nanmean(true_alert_streaks_rates_from_initial_detection)

        true_alert_rate = reshaped_predictions.mean()

        detection_rates = [undetected_samples_percentage,
                           mean_tasrid,
                           true_alert_rate]

        return detection_times, detection_rates

    @staticmethod
    def calculate_longest_consecutive_true_detection(row):
        first_true_index = np.argmax(row)  # Find the index of the first True
        longest_consecutive_true_detection = 0

        for value in row[first_true_index:]:
            if value:
                longest_consecutive_true_detection += 1
            else:
                break

        return longest_consecutive_true_detection

    @staticmethod
    def rolling_window_strider(data, window_size):
        shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
        strides = data.strides + (data.strides[-1],)
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    def detection_time_evaluator1(self, classifier, scaler):
        import time
        start_time = time.time()
        print(start_time)
        detection_times = np.zeros(self.no_tests)

        for data_index in range(self.no_tests):
            data = self.test_data[data_index]
            detection_time = None
            for i in range(self.detection_horizon):
                features = data[i:i + self.time_window]
                features = features.reshape(1, -1)
                x = scaler.transform(features)
                x = x.reshape(1, -1, 1)
                if classifier.predict(x) > 0.5:
                    detection_time = i + 1
                    break
            detection_times[data_index] = detection_time
        end_time = time.time()
        print('calculation duration: ', f"{end_time - start_time:.2f}")
        return detection_times

    def main(self):
        trained_classifiers = self.trainer()
        detection_times = {}
        detection_rates = {}
        for trainer_id in trained_classifiers.keys():
            if "mixed" in trainer_id:
                scaler = self.mixed_scaler
            else:
                scaler = self.convn_scaler

            times, rates = self.detection_time_evaluator(trained_classifiers[trainer_id], scaler)
            detection_times[trainer_id] = times
            detection_rates[trainer_id] = rates

        df_time = pd.DataFrame(detection_times)
        df_time.insert(0, 'Pattern Name', self.pattern_name)
        df_time.insert(1, 'Imbalanced Ratio', self.imbalanced_ratio)
        df_time.insert(2, 'Time Window', self.time_window)
        df_time.insert(3, 'Abnormality Value', self.abnormality_value)

        saving_directory = os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Times', self.pattern_name)
        os.makedirs(saving_directory, exist_ok=True)
        # saving_name = f"{self.time_window}_{self.abnormality_value}_{self.imbalanced_ratio}.csv"
        saving_name = f"{self.imbalanced_ratio}_{self.abnormality_value}_{self.time_window}.csv"
        df_time.to_csv(os.path.join(saving_directory, saving_name), index=False)

        metrics = ['Undetected Samples Percentage', 'Mean True Alert Streaks from Initial Detection', 'True Alert Rate']
        df_rates = pd.DataFrame(detection_rates)
        df_rates['Detection Rates Metrics'] = metrics
        df_rates.insert(0, 'Pattern Name', self.pattern_name)
        df_rates.insert(1, 'Imbalanced Ratio', self.imbalanced_ratio)
        df_rates.insert(2, 'Time Window', self.time_window)
        df_rates.insert(3, 'Abnormality Value', self.abnormality_value)
        saving_directory = os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Rates', self.pattern_name)
        os.makedirs(saving_directory, exist_ok=True)
        saving_name = f"{self.imbalanced_ratio}_{self.abnormality_value}_{self.time_window}.csv"
        df_rates.to_csv(os.path.join(saving_directory, saving_name), index=False)


if __name__ == "__main__":

    pattern_names = ["upshift", "downtrend", "uptrend", "downshift", "systematic", "stratification", "cyclic"]
    time_windows = [30, 50, 80]
    imbalanced_ratios = [0.95, 0.9, 0.75, 0.5]
    detection_horizon = 120

    abnormality_values = [0.005, 0.08, 0.155, 0.23, 0.305, 0.38, 0.455, 0.53, 0.605, 0.68, 0.755, 0.83, 0.905, 0.98,
                          1.055, 1.13, 1.205, 1.28, 1.355, 1.43, 1.505, 1.58, 1.655, 1.73, 1.805]

    for pattern_name in pattern_names:
        for imbalanced_ratio in imbalanced_ratios:
            for abnormality_value in abnormality_values:
                for time_window in time_windows:
                    experimenter = Experimenter(pattern_name, time_window, abnormality_value, imbalanced_ratio,
                                                detection_horizon)
                    experimenter.main()
