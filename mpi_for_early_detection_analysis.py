import os
import sys
import numpy as np
from itertools import product

import pandas as pd
from early_detection_analysis import Experimenter
from mpi4py import MPI

# Initialize MPI
communicator = MPI.COMM_WORLD
no_processes = communicator.Get_size()
rank = communicator.Get_rank()


def main(pattern_name):
    time_windows = [30, 50, 80]
    imbalanced_ratios = [0.95, 0.9, 0.75, 0.5]
    detection_horizon = 120

    abnormality_values = [0.005, 0.08, 0.155, 0.23, 0.305, 0.38, 0.455, 0.53, 0.605, 0.68, 0.755, 0.83, 0.905, 0.98,
                          1.055, 1.13, 1.205, 1.28, 1.355, 1.43, 1.505, 1.58, 1.655, 1.73, 1.805]

    experiments = list(product(imbalanced_ratios, abnormality_values, time_windows))

    experiments_per_process = [experiments[i] for i in range(len(experiments)) if i % no_processes == rank]
    print('done!')

    for experiment in experiments_per_process:
        imbalanced_ratio, abnormality_value, time_window = experiment
        print("Process:", rank, "- Pattern:", pattern_name, "- Experiment:", experiment)
        filename = os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Rates', pattern_name,
                                "{}_{}_{}.csv".format(str(experiment[0]), str(experiment[1]), str(experiment[2])))
        if os.path.exists(filename):
            print(filename, 'is already done!')
            continue
        
        print("NO, for ", filename)
        experimenter = Experimenter(pattern_name, time_window, abnormality_value, imbalanced_ratio, detection_horizon)
        experimenter.main()
    
    print('before concat!')

    communicator.barrier()

    if rank == 0:
        print("Pattern ", pattern_name, " experiments are completed.")

        detections_filenames = [
            os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Times', pattern_name,
                         "{}_{}_{}.csv".format(str(experiment[0]), str(experiment[1]), str(experiment[2])))
            for experiment in experiments]

        rates_filenames = [
            os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Rates', pattern_name,
                         "{}_{}_{}.csv".format(str(experiment[0]), str(experiment[1]), str(experiment[2])))
            for experiment in experiments]

        concatenated_detection_dfs = pd.concat([pd.read_csv(filename) for filename in detections_filenames],
                                               ignore_index=True)
        concatenated_rates_dfs = pd.concat([pd.read_csv(filename) for filename in rates_filenames], ignore_index=True)

        final_detection_output_filename = os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Times',
                                                       pattern_name, "{}.csv".format(pattern_name))

        final_rates_output_filename = os.path.join(os.getcwd(), 'Early Detection Analysis', 'Detection Rates',
                                                   pattern_name, "{}.csv".format(pattern_name))

        concatenated_detection_dfs.to_csv(final_detection_output_filename, index=False)
        concatenated_rates_dfs.to_csv(final_rates_output_filename, index=False)
        print("All experiments are completed. The results are saved in", final_detection_output_filename,
              'and', final_rates_output_filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mpi_for_early_detection_analysis.py <pattern_name>")
        sys.exit(1)
    pattern_name = sys.argv[1]
    main(pattern_name)
