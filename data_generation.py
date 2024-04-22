import os

import re

import json

from itertools import product

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


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
    This class generates a set of simulated datasets for the problem of control chart pattern recognition based on
    the equations in the paper of:
        An imbalance-aware BiLSTM for control chart patterns early detection, written by
            Mohammad Derakhshi and Talayeh Razzaghi.
            https://www.sciencedirect.com/science/article/abs/pii/S0957417424005487

    """
    def __init__(self, name, dataset_distribution, time_window=10, imbalanced_ratio=0.95,
                 abnormality_parameters=None, visualization=False):
        """
        Args:
            name (str): name of the generated dataset
            dataset_distribution (dict): a dictionary determining the number of data points in each set and
                subclasses:

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

        self.abnormality_rates = [1, 0.8, 0.6, 0.4, 0.2, 0]

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


class DatasetGenerator:
    """This class determines what is the best training strategy."""

    def __init__(self, pattern_name, imbalanced_ratio, dataset_distribution, no_electing_datasets):

        self.pattern_name = pattern_name
        self.imbalanced_ratio = imbalanced_ratio
        self.dataset_distribution = dataset_distribution
        self.no_electing_datasets = no_electing_datasets

    def pool_of_datasets_maker(self, saving_folder, random_parameters):
        """
        This method generates a random dataset of types self.pattern_name corresponding to each element in
        random_parameters, and save the generated dataset in the saving_folder
        Args:
            saving_folder (str): The directory where the generated datasets are stored.
            random_parameters (list): A list that its elements looks like (time_window, abnormality_parameter_value).

        Returns:
            None

        """
        for k in range(len(random_parameters)):

            file_name = self.pattern_name + "_" + str(k)
            time_window, parameter_value = random_parameters[k]

            abnormality_parameters = {'pattern_name': self.pattern_name,
                                      'parameter_value': parameter_value}

            random_data = BinaryControlChartPatterns(name=file_name,
                                                     dataset_distribution=self.dataset_distribution,
                                                     time_window=time_window,
                                                     imbalanced_ratio=self.imbalanced_ratio,
                                                     abnormality_parameters=abnormality_parameters,
                                                     visualization=True if k == 100 else False)

            saving_directory = os.path.join(os.getcwd(), saving_folder)
            if not os.path.exists(saving_directory):
                os.makedirs(saving_directory, exist_ok=True)

            saving_file_name = os.path.join(saving_directory, file_name + ".txt")
            if not os.path.isfile(saving_file_name):
                with open(saving_file_name, mode='w') as file:
                    json.dump(obj=random_data.__dict__, fp=file, cls=NumpyArrayEncoder)

                self.data_preprocessor(saving_file_name)

    def data_preprocessor(self, file_name):
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
        with open(file=file_name, mode='r') as readfile:
            file = json.load(readfile)
        raw_data = file['raw_data']

        if not ('normalized_dataset' in file.keys()):
            # concatenate all subclasses in the train set for the sake of normalization
            timeseries = np.concatenate(([raw_data['train'][class_name][0]
                                          for class_name in raw_data['train'].keys()]))

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

            output = {"normalized_dataset": normalized_dataset}

            with open(file=file_name, mode='r+') as write_file:
                data = json.load(write_file)

                data.update(output)
                write_file.seek(0)
                json.dump(data, write_file, cls=NumpyArrayEncoder)

    def random_datasets_parameters_provider(self):
        """
        This method chooses a random abnormality_paratmeter for each time_window for the purpose of generating
        random datasets that are used in model selection process. The ones that are not selected are considered
        only for calculating heatmap of the pattern_name pattern.

        Returns:
            two lists of selected_instances_for_model_selection, remaining_instances_for_heatmap

        """

        with open(file=os.path.join(os.getcwd(), "Input_data", "heatmap_scales.txt"), mode='r') as readfile:
            file = json.load(readfile)
            heatmap_scales = file[self.pattern_name]

        time_scale = heatmap_scales['time_scale']
        parameter_scale = heatmap_scales["parameter_scale"]
        selected_instances_for_model_selection = []
        remaining_instances_for_heatmap = []
        for time_window in time_scale:
            copy_parameter_scale = parameter_scale[:]
            probabilities = 0.6 - np.log(np.array(parameter_scale))
            assert all(probabilities > 0)
            probabilities = probabilities / np.sum(probabilities)
            selected_item = np.random.choice(parameter_scale, p=probabilities)
            copy_parameter_scale.remove(selected_item)
            selected_instances_for_model_selection.append((time_window, selected_item))
            remaining_items = list(product([time_window], copy_parameter_scale))
            remaining_instances_for_heatmap.extend(remaining_items)

        return selected_instances_for_model_selection, remaining_instances_for_heatmap

    def main(self):

        selected_items, remaining_items = self.random_datasets_parameters_provider()
        self.pool_of_datasets_maker('DS_for_MS', selected_items)
        self.pool_of_datasets_maker('Remaining_DS', remaining_items)


if __name__ == "__main__":

    with open(file=os.path.join(os.getcwd(), "Input_data", "instances_info.txt"), mode='r') as readfile:
        instances_info = json.load(readfile)

    pattern_names = instances_info['pattern_names']
    instances_info.pop('pattern_names')
    for pattern_name in pattern_names:
        dataset_generator = DatasetGenerator(pattern_name,
                                             imbalanced_ratio=instances_info["imbalanced_ratio"],
                                             dataset_distribution=instances_info["dataset_distribution"],
                                             no_electing_datasets=instances_info["no_electing_datasets"])
        dataset_generator.main()
