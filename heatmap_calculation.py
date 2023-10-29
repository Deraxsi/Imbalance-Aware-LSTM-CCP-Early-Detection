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

    def __init__(self, hp, strategy_name, imbalanced_ratio, pattern_name, minimum_no_epochs,
                 maximum_no_epochs, patience):
        """
        Args:
            hp (dict): best hyperparameters.
        """
        self.hp = hp

        self.is_stratified_sampling = "Stratified" in strategy_name
        self.is_cost_sensitive = "cost-sensitive" in strategy_name
        if self.is_cost_sensitive:
            self.is_equitable = not ("inequitable" in strategy_name)
        else:
            self.is_equitable = None

        self.imbalanced_ratio = imbalanced_ratio
        self.pattern_name = pattern_name

        self.minimum_no_epochs = minimum_no_epochs
        self.maximum_no_epochs = maximum_no_epochs
        self.patience = patience

        self.kappa = hp["kappa"]

        self.subclass_weights = self.subclass_weight_specifier(hp["alpha"], hp["is_emphasizing"])

        self.metric_names = ['loss', 'specificity', 'sensitivity', 'accuracy', 'geometric_mean']

    @staticmethod
    def subclass_weight_specifier(alpha, is_emphasizing):
        """this method determines the weight retaining parameters of subclasses using power function
        eta ** alpha, where eta shows the percentage of abnormal signals for a subclass."""
        subclass_weights = dict()
        for eta in [0.2, 0.4, 0.6, 0.8, 1]:
            if is_emphasizing:
                subclass_weights["abnormal_" + str(eta)] = 1 + eta ** alpha
            else:
                subclass_weights["abnormal_" + str(eta)] = 1 + (1 - eta) ** alpha

        subclass_weights["normal"] = max(subclass_weights.values())

        return subclass_weights

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

        strategy_fname = f"Strf={str(self.is_stratified_sampling)[0]}-" \
                         f"CS={str(self.is_cost_sensitive)[0]}-" \
                         f"EQ={str(self.is_stratified_sampling)[0]}"

        curves_paths = os.path.join(os.getcwd(), 'Offline_Learning_Curves', 'In_heatmaps', self.pattern_name,
                                    strategy_fname)
        curves_save_fname = os.path.join(curves_paths, time_param + '.png')

        os.makedirs(curves_paths, exist_ok=True)

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
                            # callbacks=[gmean_callback, history_callback],
                            callbacks=[gmean_callback],
                            validation_data=valid_data,
                            validation_batch_size=valid_batch_size,
                            verbose=0)

        trained_classifier = history.model

        # self.learning_curve_visualizer(history, time_param)

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

        # print(f'The training of {file_directory} file is just finished!')
        return (time_window, abnormality_value, imbalanced_test_results, balanced_test_results)

    def heatmap_calculator(self):
        """
        Returns:
            heatmap_data (dict): {(time_window, abnormality_parameter_value): best_tuner performance of the
                                  related dataset}
            metric_names (keys): a dictionary keys containing 'loss', 'specificity', 'sensitivity',', accuracy',
                                 and 'geometric_mean'

        """
        imbalanced_heatmap_data = {}
        balanced_heatmap_data = {}

        ms_datasets_path = os.path.join(os.getcwd(), 'DS_for_MS', self.pattern_name + '_*.txt')
        dataset_directories = glob.glob(ms_datasets_path)
        remaining_datasets_path = os.path.join(os.getcwd(), 'Remaining_DS', self.pattern_name + '_*.txt')
        remaining_files = glob.glob(remaining_datasets_path)
        dataset_directories.extend(remaining_files)

        assert len(dataset_directories) == 250

        # # todo: remove this part! for test only!
        results = []
        for dataset_directory in dataset_directories:
            result = self.fit_per_dataset(dataset_directory)
            results.append(result)

        # with ProcessPoolExecutor(max_workers=1) as executor:  # todo:
        #
        #     futures = [executor.submit(self.fit_per_dataset, dataset_directory)
        #                for dataset_directory in dataset_directories]  # todo:
        #     results = [future.result() for future in futures]

        for result in results:
            imbalanced_heatmap_data[(result[0], result[1])] = dict(zip(self.metric_names, np.array(result[2])))
            balanced_heatmap_data[(result[0], result[1])] = dict(zip(self.metric_names, np.array(result[3])))

        return imbalanced_heatmap_data, balanced_heatmap_data

    def heatmap_visulizer(self, data, is_balanced):
        """
        This method visualizes the heatmaps.
        Args:
            data (dict): {(time_window, abnormality_parameter_value): best_tuner performance of the
                                  related dataset}
        Returns:
            None

        """
        df = pd.DataFrame.from_dict(data, orient='index')
        reindexed_df = df.reset_index()
        renamed_df = reindexed_df.rename(columns={'level_0': 'Time window', 'level_1': 'Abnormality parameter'})

        for metric_name in self.metric_names:
            if not (metric_name == "loss"):
                lower_range = 0
                upper_range = 1
            else:
                lower_range = None
                upper_range = None

            heatmap_data = renamed_df.pivot_table(values=metric_name, index='Time window',
                                                  columns='Abnormality parameter', aggfunc=np.sum)
            heatmap_data = heatmap_data.sort_index(ascending=False)
            heatmap_data = heatmap_data.sort_index(1)

            plt.figure()
            # sb.heatmap(heatmap_data, linewidths=1, annot=False, cmap='coolwarm', vmin=lower_range, vmax=upper_range)
            sb.heatmap(heatmap_data, linewidths=0, annot=False, cmap='YlGnBu', vmin=lower_range, vmax=upper_range)

            strategy_fname = f"Strf={str(self.is_stratified_sampling)[0]}-" \
                             f"CS={str(self.is_cost_sensitive)[0]}-" \
                             f"EQ={str(self.is_stratified_sampling)[0]}"
            heatmap_directory = os.path.join(os.getcwd(), 'Offline_Heatmaps', strategy_fname)
            os.makedirs(heatmap_directory, exist_ok=True)

            heatmap_results_directory = os.path.join(os.getcwd(), 'Offline_Heatmaps_Results', strategy_fname)
            os.makedirs(heatmap_results_directory, exist_ok=True)

            if is_balanced:
                fig_name = 'balanced_' + self.pattern_name + '_' + metric_name + '.png'
                csv_fname = 'balanced_' + self.pattern_name + '_' + metric_name + '.csv'
            else:
                fig_name = 'imbalanced_' + self.pattern_name + '_' + metric_name + '.png'
                csv_fname = 'imbalanced_' + self.pattern_name + '_' + metric_name + '.csv'

            plt.savefig(os.path.join(heatmap_directory, fig_name))

            saving_fname = os.path.join(heatmap_results_directory, csv_fname)
            pd.DataFrame.to_csv(heatmap_data, saving_fname, header=True, index=True)
            plt.close()


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
        self.best_loss = 10**10
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

    def main(self):
        with open(file=os.path.join(os.getcwd(), "Input_data", "instances_info.txt"), mode='r') as readfile:
            instances_info = json.load(readfile)

        with open(file=os.path.join(os.getcwd(), "Input_data",
                                    "offline_classification_parameters.txt"), mode='r') as readfile:
            classification_parameters = json.load(readfile)

        # pattern_names = instances_info['pattern_names']
        pattern_names = ['uptrend']
        imbalanced_ratio = instances_info["imbalanced_ratio"]

        for pattern_name in pattern_names:
            best_hp_directory = os.path.join(os.getcwd(),
                                             "Offline_Best_Models",
                                             "%.2f" % imbalanced_ratio + "_imbalanced_ratio", "BayesianOptimization",
                                             pattern_name + ".txt")
            with open(file=best_hp_directory, mode='r') as readfile:
                best_models = json.load(readfile)

            for strategy_name in best_models.keys():
                best_model_per_strategy = best_models[strategy_name]

                classifiers = Classifiers(hp=best_model_per_strategy,
                                          imbalanced_ratio=imbalanced_ratio,
                                          strategy_name=strategy_name,
                                          pattern_name=pattern_name,
                                          minimum_no_epochs=classification_parameters["minimum_no_epochs"],
                                          maximum_no_epochs=classification_parameters["maximum_no_epochs"],
                                          patience=classification_parameters["patience"])

                imbalanced_heatmap_data, balanced_heatmap_data = classifiers.heatmap_calculator()
                classifiers.heatmap_visulizer(imbalanced_heatmap_data, is_balanced=False)
                classifiers.heatmap_visulizer(balanced_heatmap_data, is_balanced=True)


if __name__ == "__main__":
    experimenter = Experimenter()
    experimenter.main()
