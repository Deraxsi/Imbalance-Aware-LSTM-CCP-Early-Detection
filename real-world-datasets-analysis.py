import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import Model, Input

from sklearn.utils import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import LearningRateScheduler


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


def scheduler(epoch, lr, kappa=0.1):
    new_lr = lr * kappa
    return new_lr


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


class RealWorldCCPRs:
    def __init__(self):
        pass

    def pharmaceutical_data_visualizer(self, df, selected_vars, saving_directory):

        # plot correction matrix------------------
        corr_matrix = df[selected_vars].corr()
        plt.figure(figsize=(10, 8))

        threshold = 0.5
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)
        sns.heatmap(corr_matrix_masked, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Among Output Variables')
        plt.subplots_adjust(bottom=0.25)
        plt.xlabel('Output Variables')
        plt.ylabel('Output Variables')
        plt.xticks(rotation=45, ha='right')

        plt.savefig(os.path.join(saving_directory, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # missing values and visualization------------------------------------
        missing_values = df.isnull().sum()
        plt.figure(figsize=(10, 6))
        plt.bar(missing_values.index, missing_values.values)
        plt.xlabel('Output Variables')
        plt.ylabel('Missing Values')
        plt.title('Missing Values in Dataset')

        # Rotate x-axis tick labels for better readability (optional)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(saving_directory, 'missingness.png'))
        plt.close()

        missing_percentages = (df[selected_vars].isnull().sum() / len(df)) * 100
        missing_color = '#E5E5E5'
        available_color = '#4682B4'

        plt.figure(figsize=(10, 6))
        plt.bar(missing_percentages.index, missing_percentages, color=missing_color)

        for i, val in enumerate(missing_percentages):
            if val < 100:  # Assuming 100% missing values means all values are missing
                plt.bar(i, 100 - val, bottom=val, color=available_color)

        plt.title('Missing Values Per Output Variable')
        plt.xlabel('Variables')
        plt.ylabel('Missing Percentage')
        plt.subplots_adjust(bottom=0.25)
        plt.xticks(rotation=45, ha='right')
        for i, val in enumerate(missing_percentages):
            plt.text(i, 100, f'{val:.1f}%', ha='center', va='bottom')
        plt.savefig(os.path.join(saving_directory, 'missingness2.png'))
        plt.close()

        # titer_per_vessel visualization -------------------------------
        filtered_df = df[df['Production Day'] == 14]
        sorted_df = filtered_df.sort_values(by='Titer', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        color_code_1 = '#4682B4'
        color_code_2 = '#FF6347'
        alphas = [1, 0.9, 0.9, 0.9, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9,
                  0.9, 0.9, 1, 1]

        colors = [color_code_1] * 6 + [color_code_2] * (len(alphas) - 6)

        for i, titer_value in enumerate(sorted_df['Titer']):
            ax.bar(i, titer_value, color=colors[i], alpha=alphas[i])

        # plt.title('Titer for Each Vessel on Production Day 14')
        plt.xlabel('Vessel Names', fontsize=14)
        plt.ylabel('Titer', fontsize=14)
        plt.xticks(range(len(sorted_df)), sorted_df['Vessel Name'], rotation=45, ha='right', color='black')
        plt.subplots_adjust(bottom=0.25)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(saving_directory, 'titer_per_vessel.png'))
        plt.close()

        vessel_names = sorted_df['Vessel Name']

        alphas = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.6, 0.7,
                  0.8, 0.9, 0.9, 1]

        # Create a figure and axis
        # Plot the time series of 'Variable 2' for each vessel name with varying transparency ----------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (vessel_name, alpha) in enumerate(zip(vessel_names, alphas)):
            if i <= 6:
                color = color_code_1
            else:
                color = color_code_2

            # vessel_data = df[(df['Vessel Name'] == vessel_name) & (df['Production Day'] >= 7)]
            vessel_data = df[df['Vessel Name'] == vessel_name]
            ax.plot(vessel_data['Production Day'], vessel_data['Density'], alpha=alpha, color=color,
                    label=vessel_name)
            # last_data_point = vessel_data.iloc[-1]
            # ax.annotate(vessel_name, xy=(last_data_point['Production Day'], last_data_point['Density']),
            #             xytext=(-20, 5), textcoords='offset points', color=color)

        # ax.set_title('Monitoring Viable Cell Density per Vessel Name During 2nd Production Week')
        ax.set_xlabel('Production Day', fontsize=16)
        ax.set_ylabel('Viable Cell Density', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Vessel Names', ncol=3)
        # legend = ax.legend(title='Vessel Names', ncol=4)
        # frame = legend.get_frame()
        # frame.set_edgecolor('black')
        plt.tight_layout()

        plt.savefig(os.path.join(saving_directory, 'viable-density.png'))

    def pharmaceutical_data_provider(self, do_visualization=True):

        df = pd.read_excel(os.path.abspath(os.path.join(os.getcwd(), "..\\..", 'Cytovance', "CHOKC Data for OU.xlsx")),
                           sheet_name='CHOKC Seed & feed - readable')
        variables = df.columns.tolist()
        selected_vars = variables[:][2:]
        df['Vessel Name'] = df['Vessel Name'].apply(lambda x: f'Vsl. {x}')

        saving_derctory = os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical')
        os.makedirs(saving_derctory, exist_ok=True)
        if do_visualization:
            self.pharmaceutical_data_visualizer(df, selected_vars, saving_derctory)
            df = df.dropna(axis=1)
        else:
            df = df.dropna(axis=1)

        # raw_data = df[['Vessel Name', 'Production Day', 'Density']]
        desirable_patterns = [f'Vsl. {x}' for x in [16, 23, 4, 20, 8, 11]]
        pivot_df = df.pivot(index='Vessel Name', columns='Production Day', values='Density')
        pivot_df = pivot_df.reset_index()
        pivot_df['Label'] = pivot_df["Vessel Name"].map(lambda x: 0 if x in desirable_patterns else 1)

        new_column_names = {i: f'Day {i + 1}' for i in range(len(pivot_df.columns[1:-1]))}
        revised_df = pivot_df.rename(columns=new_column_names)

        # rdf = pivot_df.iloc[:, [0] + list(range(8, len(pivot_df.columns)))]
        # rdf = pivot_df.loc[:, [rdf.columns[-1]] + rdf.columns[1:-1]]
        revised_df.to_csv(os.path.join(saving_derctory, 'pharmaceutical.csv'), index=False)

        revised_df.drop('Vessel Name', axis=1, inplace=True)

        return revised_df

    @staticmethod
    def calculate_metrics(true_labels, predicted_labels):
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[False, True]).ravel()

        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        gmean = np.sqrt(specificity * sensitivity)

        output = dict(zip(['Specificity', 'Sensitivity', 'Accuracy', 'G-mean'],
                          [specificity, sensitivity, accuracy, gmean]))

        return output

    # this one is used when I wanted to train pharmaceutical data with low number of rollings!
    # @staticmethod
    # def rolling_window_strider(features, labels, window_size):
    #
    #     shape = features.shape[:-1] + (features.shape[-1] - window_size + 1, window_size)
    #     strides = features.strides + (features.strides[-1],)
    #     rolled_features = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)
    #
    #     rolled_features = rolled_features.reshape((-1, window_size))
    #     no_rollings = features.shape[-1] - window_size + 1
    #
    #     rolled_labels = np.repeat(labels, no_rollings)
    #
    #     return rolled_features, rolled_labels, no_rollings

    @staticmethod
    def rolling_window_strider(features, labels, window_size, stride=1):
        shape = features.shape[:-1] + ((features.shape[-1] - window_size) // stride + 1, window_size)
        strides = features.strides[:-1] + (stride * features.strides[-1], features.strides[-1])
        rolled_features = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)

        rolled_features = rolled_features.reshape((-1, window_size))
        no_rollings = (features.shape[-1] - window_size) // stride + 1

        rolled_labels = np.repeat(labels, no_rollings)

        return rolled_features, rolled_labels, no_rollings

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

    def build(self, hp, time_window):
        """
        Args:
            hp (dict): dictionary containing information of best model's architecture.
            time_window(int): the value of time_window with which the classifier should be constructed accordingly.
        """
        input_layer = Input(shape=(time_window, 1))
        no_layers = hp["no_layers"]
        x = input_layer
        for layer in range(1, no_layers + 1):
            shared_names = f'_in_layer_{layer}'

            no_units = hp[f'time_window={time_window}/no_units' + shared_names]
            dropout = hp[f'time_window={time_window}/dropout' + shared_names]

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

        learning_rate = hp['learning_rate']
        optimizer = Nadam(learning_rate, clipvalue=0.5, clipnorm=1.0)

        metrics = [
            Specificity(),
            Sensitivity(),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            GeometricMean()]

        loss = FocalLoss(gamma=2, label_smoothing=0)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)

        return model

    def tuned_hps_provider(self, dataset_name='pharmaceutical'):
        if dataset_name == 'pharmaceutical':
            pattern_name = 'downtrend'
        elif dataset_name == 'wafer':
            pattern_name = 'downshift'
        else:
            raise ValueError(f"{dataset_name} is not found!")

        best_hp_directory = os.path.join(os.getcwd(),
                                         "Offline_Best_Models",
                                         "0.95_imbalanced_ratio",
                                         "BayesianOptimization",
                                         pattern_name + ".txt")
        with open(file=best_hp_directory, mode='r') as readfile:
            best_models = json.load(readfile)

        selected_strategy = "Stratified sampling, with cost-sensitive loss-function and inequitable sub-class weighting"
        tuned_parameters = best_models[selected_strategy]

        return tuned_parameters

    def pharmaceutical_classifier_fit(self, df, time_window=10):

        tuned_hps = self.tuned_hps_provider(dataset_name='pharmaceutical')

        data = df.values
        input_sequences = data[:, :-1]
        labels = data[:, -1]

        epochs = list(range(15, 80, 1))

        for epoch in epochs:

            kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=117)

            metrics_df = pd.DataFrame(columns=['Fold ID', 'Rolling ID', 'Specificity', 'Sensitivity', 'Accuracy',
                                               'G-mean'])

            detection_time_df = pd.DataFrame(columns=['Fold ID',
                                                      'Average Run Length (Positives)',
                                                      'Average Run Length (Negatives)',
                                                      'Undetected Percentage (Positives)',
                                                      'Undetected Percentage (Positives)',
                                                      "True Early Alert Streaks (Positives)",
                                                      "True Early Alert Streaks (Negatives)",
                                                      "True Alert Rate (Positives)",
                                                      "True Alert Rate (Negatives)"])

            for fold_id, (train_index, test_index) in enumerate(kfold.split(input_sequences, labels)):
                scaler = StandardScaler()

                # split data into train and test
                x_train, x_test = input_sequences[train_index], input_sequences[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                # do rolling window on train and test
                x_train_rolled, y_train_rolled, no_rolls_train = self.rolling_window_strider(x_train, y_train,
                                                                                             time_window)
                x_test_rolled, y_test_rolled, no_rolls_test = self.rolling_window_strider(x_test, y_test, time_window)

                assert no_rolls_train == no_rolls_test

                normalized_train = scaler.fit_transform(x_train_rolled)

                normalized_train = normalized_train.reshape(-1, no_rolls_train, time_window)
                normalized_train = normalized_train.reshape(-1, time_window, 1)

                model = self.build(tuned_hps, time_window)

                sample_weights = compute_sample_weight('balanced', y_train_rolled)
                y_train_rolled = y_train_rolled.astype(float)
                model.fit(normalized_train, y_train_rolled, epochs=epoch, verbose=1, sample_weight=sample_weights)

                # calculate metrics on test
                normalized_test = scaler.transform(x_test_rolled)
                normalized_test = normalized_test.reshape(-1, no_rolls_test, time_window)
                normalized_test = normalized_test.reshape(-1, time_window, 1)

                predicted_labels = model.predict(normalized_test) > 0.5
                true_labels = y_test_rolled > 0.5

                predicted_labels_reshaped = predicted_labels.reshape(-1, no_rolls_test)
                true_labels_reshaped = true_labels.reshape(-1, no_rolls_test)

                for rolling_id in range(no_rolls_test):
                    metrics = self.calculate_metrics(true_labels_reshaped[:, rolling_id],
                                                     predicted_labels_reshaped[:, rolling_id])
                    metrics['Rolling ID'] = rolling_id
                    metrics['Fold ID'] = fold_id

                    metrics_df.loc[len(metrics_df)] = metrics

                # detection time and no of failures
                positive_rows = np.all(true_labels_reshaped, axis=1)
                negative_rows = np.all(~true_labels_reshaped, axis=1)

                matching_mask = predicted_labels_reshaped == true_labels_reshaped
                detection_indices = np.argmax(matching_mask, axis=1).astype(float)
                detection_indices[~np.any(matching_mask, axis=1)] = np.nan

                detection_times = np.copy(detection_indices)
                detection_times[~np.isnan(detection_times)] += 1

                matching_positives = matching_mask[positive_rows]
                matching_negatives = matching_mask[negative_rows]

                arl_positivse = detection_times[positive_rows].mean()
                arl_negatives = detection_times[negative_rows].mean()
                undetected_percentage_positivse = np.isnan(detection_times[positive_rows]).sum() / positive_rows.sum()
                undetected_percentage_negatives = np.isnan(detection_times[negative_rows]).sum() / negative_rows.sum()

                early_alert_streaks_rates_positives = np.apply_along_axis(
                    self.calculate_longest_consecutive_true_detection,
                    axis=1,
                    arr=matching_positives) / no_rolls_test
                early_alert_streaks_rates_negatives = np.apply_along_axis(
                    self.calculate_longest_consecutive_true_detection,
                    axis=1,
                    arr=matching_negatives) / no_rolls_test

                mean_early_alert_steaks_rates_positives = early_alert_streaks_rates_positives.mean()
                mean_early_alert_steaks_rates_negatives = early_alert_streaks_rates_negatives.mean()

                true_alert_rates_positives = matching_positives.sum() / (no_rolls_test * positive_rows.sum())
                true_alert_rates_negatives = matching_negatives.sum() / (no_rolls_test * negative_rows.sum())

                new_entry = {'Fold ID': fold_id,
                             'Average Run Length (Positives)': arl_positivse,
                             'Average Run Length (Negatives)': arl_negatives,
                             'Undetected Percentage (Positives)': undetected_percentage_positivse,
                             'Undetected Percentage (Negatives)': undetected_percentage_negatives,
                             "True Early Alert Streaks (Positives)": mean_early_alert_steaks_rates_positives,
                             "True Early Alert Streaks (Negatives)": mean_early_alert_steaks_rates_negatives,
                             "True Alert Rate (Positives)": true_alert_rates_positives,
                             "True Alert Rate (Negatives)": true_alert_rates_negatives}

                detection_time_df.loc[len(detection_time_df)] = new_entry

            metrics_df.to_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                           f'classification_metrics_{epoch}.csv'),
                              index=False)

            detection_time_df.to_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                                  f'detection_metrics_{epoch}.csv'), index=False)

        classification_metrics_dfs = []
        detection_metrics_dfs = []
        for epoch in epochs:
            classification_metrics_df = pd.read_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                                                 f'classification_metrics_{epoch}.csv'))
            classification_metrics_df.insert(loc=len(classification_metrics_df.columns), column='epoch', value=epoch)

            classification_metrics_dfs.append(classification_metrics_df)

            detection_metrics_df = pd.read_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                                            f'detection_metrics_{epoch}.csv'))
            detection_metrics_df.insert(loc=len(detection_metrics_df.columns), column='epoch', value=epoch)
            detection_metrics_dfs.append(detection_metrics_df)

        all_classification_metrics_df = pd.concat(classification_metrics_dfs)
        all_detection_time_df = pd.concat(detection_metrics_dfs)

        all_classification_metrics_df.to_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                             f'classification_metrics_ALL.csv'), index=False)

        all_detection_time_df.to_csv(os.path.join(os.getcwd(), "Real-world Cases", 'Pharmaceutical',
                                                  f'detection_metrics_ALL.csv'), index=False)

    def wafer_data_provider(self):
        train_df = pd.read_csv(os.path.join(os.getcwd(), "UCRArchive_2018", "Wafer", "Wafer_TRAIN.tsv"),
                               sep='\t',
                               header=None)
        test_df = pd.read_csv(os.path.join(os.getcwd(), "UCRArchive_2018", "Wafer", "Wafer_TEST.tsv"),
                              sep='\t',
                              header=None)

        train_df.iloc[:, 0] = train_df.iloc[:, 0].replace(-1, 0)
        test_df.iloc[:, 0] = test_df.iloc[:, 0].replace(-1, 0)

        train_df = train_df.iloc[:, :-2]
        test_df = test_df.iloc[:, :-2]

        print("\nMissing Values:")
        print(train_df.isnull().sum().sum())
        print(test_df.isnull().sum().sum())

        saving_directory = os.path.join(os.getcwd(), "Real-world Cases", "Wafer")
        os.makedirs(saving_directory, exist_ok=True)

        train_df.to_csv(os.path.join(saving_directory, "trn.csv"), header=None, index=False)
        test_df.to_csv(os.path.join(saving_directory, "tst.csv"), header=None, index=False)

        for set_name, data_df in [('train', train_df), ('test', test_df)]:
            label_counts = data_df.iloc[:, 0].value_counts()
            label_colors = {0: '#4682B4', 1: '#FF6347'}
            label_names = {0: "Normal", 1: "Abnormal"}
            fig, ax = plt.subplots(figsize=(5, 3))

            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # sns.barplot(x=label_counts.index, y=label_counts.values, palette=label_colors, ax=ax, width=0.4)
            sns.barplot(x=label_counts.index, y=label_counts.values, palette=label_colors, ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Size")
            ax.set_xticklabels([label_names[label] for label in label_counts.index])
            ax.tick_params(axis='both', which='both', length=0)
            for index, value in enumerate(label_counts.values):
                ax.text((index+1) % 2, value, str(value), ha='center', va='bottom', fontsize=10, color='black')
            sns.despine()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(saving_directory, f"dist_{set_name}.png"))
            plt.close()
            print('done')

        for set_name, data_df in [('train', train_df), ('test', test_df)]:
            label_colors = {0: '#4682B4', 1: '#FF6347'}
            label_names = {0: "Normal", 1: "Abnormal"}

            plt.figure(figsize=(10, 6))
            plt.title(f"Time Series Instances - {set_name}", fontsize=16)

            for class_label in [0, 1]:
                class_data = data_df[data_df.iloc[:, 0] == class_label].iloc[:, 1:]
                class_data_samples = class_data.sample(6)  # Select five instances

                for index, row in class_data_samples.iterrows():
                    plt.plot(row, label=f"{label_names[class_label]} - Instance {index}",
                             color=label_colors[class_label])

            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            sns.despine()
            plt.tight_layout()
            plt.savefig(os.path.join(saving_directory, f"series_{set_name}.png"))
            # plt.show()

        return train_df, test_df

    def wafer_classifier_fit(self, train_df, test_df, time_window=100, stride=5):

        train_data = train_df.values
        x_train = train_data[:, 1:]
        y_train = train_data[:, 0]

        train_indices, valid_indices = train_test_split(np.arange(y_train.shape[0]),
                                                        train_size=0.8,
                                                        stratify=y_train,
                                                        random_state=42)

        train_features = x_train[train_indices, :]
        valid_features = x_train[valid_indices, :]
        train_labels = y_train[train_indices]
        valid_labels = y_train[valid_indices]

        test_data = test_df.values
        test_features = test_data[:, 1:]
        test_labels = test_data[:, 0]

        # do rolling window with step length of 5
        x_train_rolled, y_train_rolled, no_rolls_train = self.rolling_window_strider(train_features,
                                                                                     train_labels,
                                                                                     window_size=time_window,
                                                                                     stride=stride)

        x_valid_rolled, y_valid_rolled, no_rolls_valid = self.rolling_window_strider(valid_features,
                                                                                     valid_labels,
                                                                                     window_size=time_window,
                                                                                     stride=stride)

        x_test_rolled, y_test_rolled, no_rolls_test = self.rolling_window_strider(test_features,
                                                                                  test_labels,
                                                                                  window_size=time_window,
                                                                                  stride=stride)
        # do normalization
        scaler = StandardScaler()
        normalized_train = scaler.fit_transform(x_train_rolled)
        normalized_valid = scaler.transform(x_valid_rolled)
        normalized_test = scaler.transform(x_test_rolled)

        normalized_valid = normalized_valid.reshape(-1, no_rolls_valid, time_window)
        normalized_valid = normalized_valid.reshape(-1, time_window, 1)

        normalized_test = normalized_test.reshape(-1, no_rolls_test, time_window)
        normalized_test = normalized_test.reshape(-1, time_window, 1)

        tuned_hps = self.tuned_hps_provider(dataset_name='wafer')
        model = self.build(tuned_hps, time_window=time_window)

        gmean_callback = MultiObjectiveEarlyStopping(patience=250,
                                                     kappa=1.8,
                                                     minimum_no_epochs=5,
                                                     verbosity=1)

        y_train_rolled = y_train_rolled.astype(float)
        generator = DataGenerator(normalized_train, y_train_rolled, 800)
        model.fit(generator, epochs=1000, verbose=1, validation_data=(normalized_valid, y_valid_rolled),
                  validation_batch_size=y_valid_rolled.shape[0], callbacks=[gmean_callback])

        # evaluation on test and metrics calculations
        predicted_labels = model.predict(normalized_test) > 0.5
        true_labels = y_test_rolled > 0.5

        predicted_labels_reshaped = predicted_labels.reshape(-1, no_rolls_test)
        true_labels_reshaped = true_labels.reshape(-1, no_rolls_test)

        # classification_metrics = pd.DataFrame(columns=['Rolling ID', 'Specificity', 'Sensitivity', 'Accuracy',
        #                                                'G-mean'])
        classification_metrics = pd.DataFrame(columns=['Rolling ID', 'Specificity', 'Sensitivity', 'Accuracy',
                                                       'G-mean', 'Time Window', 'Stride'])
        for rolling_id in range(no_rolls_test):
            metrics = self.calculate_metrics(true_labels_reshaped[:, rolling_id],
                                             predicted_labels_reshaped[:, rolling_id])

            metrics['Rolling ID'] = rolling_id
            metrics['Time Window'] = time_window
            metrics['Stride'] = stride
            classification_metrics.loc[len(classification_metrics)] = metrics

        # detection time and no of failures
        positive_rows = np.all(true_labels_reshaped, axis=1)
        negative_rows = np.all(~true_labels_reshaped, axis=1)

        matching_mask = predicted_labels_reshaped == true_labels_reshaped
        detection_indices = np.argmax(matching_mask, axis=1).astype(float)
        # detection_indices = time_window + np.argmax(matching_mask, axis=1).astype(float) * stride
        detection_indices[~np.any(matching_mask, axis=1)] = np.nan

        detection_times = np.copy(detection_indices)
        detection_times[~np.isnan(detection_times)] *= stride
        detection_times[~np.isnan(detection_times)] += time_window

        matching_positives = matching_mask[positive_rows]
        matching_negatives = matching_mask[negative_rows]

        arl_positivse = np.nanmean(detection_times[positive_rows])
        arl_negatives = np.nanmean(detection_times[negative_rows])

        undetected_percentage_positivse = np.isnan(detection_times[positive_rows]).sum() / positive_rows.sum()
        undetected_percentage_negatives = np.isnan(detection_times[negative_rows]).sum() / negative_rows.sum()

        early_alert_streaks_rates_positives = np.apply_along_axis(
            self.calculate_longest_consecutive_true_detection,
            axis=1,
            arr=matching_positives) / no_rolls_test
        early_alert_streaks_rates_negatives = np.apply_along_axis(
            self.calculate_longest_consecutive_true_detection,
            axis=1,
            arr=matching_negatives) / no_rolls_test

        mean_early_alert_steaks_rates_positives = early_alert_streaks_rates_positives.mean()
        mean_early_alert_steaks_rates_negatives = early_alert_streaks_rates_negatives.mean()

        true_alert_rates_positives = matching_positives.sum() / (no_rolls_test * positive_rows.sum())
        true_alert_rates_negatives = matching_negatives.sum() / (no_rolls_test * negative_rows.sum())

        detection_metrics = {'Average Run Length (Positives)': arl_positivse,
                             'Average Run Length (Negatives)': arl_negatives,
                             'Undetected Percentage (Positives)': undetected_percentage_positivse,
                             'Undetected Percentage (Negatives)': undetected_percentage_negatives,
                             "True Early Alert Streaks (Positives)": mean_early_alert_steaks_rates_positives,
                             "True Early Alert Streaks (Negatives)": mean_early_alert_steaks_rates_negatives,
                             "True Alert Rate (Positives)": true_alert_rates_positives,
                             "True Alert Rate (Negatives)": true_alert_rates_negatives,
                             "Time Window": time_window,
                             "Stride": stride}

        detection_metrics_df = pd.DataFrame(detection_metrics, index=[0])

        saving_directory = os.path.join(os.getcwd(), "Real-world Cases", 'Wafer')
        os.makedirs(saving_directory, exist_ok=True)

        classification_metrics.to_csv(os.path.join(saving_directory,
                                                   f'classification_metrics_{time_window}_{stride}.csv'), index=False)

        detection_metrics_df.to_csv(os.path.join(saving_directory,
                                                 f'detection_metrics_{time_window}_{stride}.csv'), index=False)

        print('done')


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size

        self.no_datapoints = labels.shape[0]

        self.batched_data = self.batch_data_provider()

    def batch_size_checker(self):
        if self.no_datapoints % self.batch_size:
            raise ValueError(f"Please provide a batch size that is dividable by {self.batch_size}")

    def __len__(self):
        return np.ceil(self.no_datapoints / self.batch_size).astype(int)

    def __getitem__(self, batch_id):
        return self.batched_data[batch_id]

    def batch_data_provider(self):
        batched_data = list()
        n_splits = np.ceil(self.no_datapoints / self.batch_size).astype(int)
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=self.batch_size, random_state=42)

        for train_index, test_index in sss.split(self.features, self.labels):
            x_mini_batch, y_mini_batch = self.features[test_index], self.labels[test_index]
            sw_mini_batch = compute_sample_weight('balanced', y_mini_batch)
            rehsaped_x_mini_batch = x_mini_batch.reshape((x_mini_batch.shape[0], -1, 1))

            batched_data.append((rehsaped_x_mini_batch, y_mini_batch, sw_mini_batch))

        return batched_data


if __name__ == "__main__":
    realworld_ccprs = RealWorldCCPRs()
    # for pharama
    # df = realworld_ccprs.pharmaceutical_data_provider()
    # realworld_ccprs.pharmaceutical_classifier_fit(df)

    # for wafer
    train_df, test_df = realworld_ccprs.wafer_data_provider()
    time_windows = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    strides = [10, 5, 1]
    saving_directory = os.path.join(os.getcwd(), "Real-world Cases", 'Wafer')
    metrics_dfs = []
    detection_dfs = []
    for time_window in time_windows:
        for stride in strides:
            print("-------time_window, ", time_window, ' stride, ', stride)
            realworld_ccprs.wafer_classifier_fit(train_df, test_df, time_window, stride)

            metrics_df = pd.read_csv(os.path.join(saving_directory,
                                                  f'classification_metrics_{time_window}_{stride}.csv'))
            metrics_dfs.append(metrics_df)

            detection_df = pd.read_csv(os.path.join(saving_directory,
                                                    f'detection_metrics_{time_window}_{stride}.csv'))
            detection_dfs.append(detection_df)

    joined_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
    joined_metrics_df.to_csv(os.path.join(saving_directory, f'classification_metrics_ALL.csv'), index=False)

    joined_detection_df = pd.concat(detection_dfs, ignore_index=True)
    joined_detection_df.to_csv(os.path.join(saving_directory, f'detection_metrics_ALL.csv'), index=False)

    print('done')
