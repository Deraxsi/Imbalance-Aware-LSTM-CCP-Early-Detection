import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator


def heatmap_redrawer(subdirectory, pattern_name):
    csv_file = os.path.join(os.getcwd(),
                            "Offline_Heatmaps_Results",
                            subdirectory,
                            f'balanced_{pattern_name}_geometric_mean.csv')
    df = pd.read_csv(csv_file, index_col="Time window")

    plt.figure()
    sb.heatmap(df, linewidths=0, annot=False, cmap='YlGnBu', vmin=0, vmax=1)
    os.makedirs(os.path.join(os.getcwd(), 'Redrawn-Heatmaps'), exist_ok=True)
    plt.ylabel('Window Length', fontsize=14)
    plt.xlabel('Abnormality Value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    save_directory = os.path.join(os.getcwd(), 'Redrawn-Heatmaps', subdirectory)
    os.makedirs(save_directory, exist_ok=True)
    plt.savefig(os.path.join(save_directory, 'balanced_' + pattern_name + '_geometric_mean.png'))

    plt.close()
    print()


def heatmap_analyzer():
    pattern_names = ["uptrend", "downtrend", "upshift", "downshift", "systematic", 'cyclic', "stratification"]

    lstm_wins_percentages = []
    cslstm_wins_percentages = []
    draws_percentages = []
    for pattern_name in pattern_names:
        path_lstm = os.path.join(os.getcwd(),
                                 "Offline_Heatmaps_Results",
                                 'Strf=T-CS=F-EQ=T',
                                 f'balanced_{pattern_name}_geometric_mean.csv')

        path_cslstm = os.path.join(os.getcwd(),
                                   "Offline_Heatmaps_Results",
                                   'Strf=T-CS=T-EQ=T',
                                   f'balanced_{pattern_name}_geometric_mean.csv')

        lstm_heatmap = pd.read_csv(path_lstm, index_col="Time window").to_numpy()
        cslstm_heatmap = pd.read_csv(path_cslstm, index_col="Time window").to_numpy()

        lstm_wins_mask = lstm_heatmap > cslstm_heatmap
        cslstm_wins_mask = lstm_heatmap < cslstm_heatmap
        equality_maks = lstm_heatmap == cslstm_heatmap

        no_instances = lstm_heatmap.size

        lstm_wins_percentage_per_pattern = (lstm_wins_mask.sum().sum() / no_instances) * 100
        cslstm_wins_percentage_per_pattern = (cslstm_wins_mask.sum().sum() / no_instances) * 100
        draws_percentage_per_pattern = (equality_maks.sum().sum() / no_instances) * 100

        lstm_wins_percentages.append(lstm_wins_percentage_per_pattern)
        cslstm_wins_percentages.append(cslstm_wins_percentage_per_pattern)
        draws_percentages.append(draws_percentage_per_pattern)

    wins_comparer = {'Pattern': pattern_names,
                     'BiLSTM - MRDS': lstm_wins_percentages,
                     'CSBiLSTM - MRDS': cslstm_wins_percentages,
                     'Draws': draws_percentages}

    df = pd.DataFrame(wins_comparer)

    # Set the style
    fig, ax = plt.subplots(figsize=(6, 5))
    # sb.set(style="whitegrid")

    patterns = df['Pattern'].unique()
    x = range(len(patterns))  # X-axis points

    bi_lstm_mrds = df["BiLSTM - MRDS"].values
    cs_bi_lstm_mrds = df["CSBiLSTM - MRDS"].values
    draws = df["Draws"].values

    plt.bar(x, bi_lstm_mrds, color='#A6CEE3', width=0.6, label='BiLSTM - MRDS Wins')
    plt.bar(x, cs_bi_lstm_mrds, bottom=bi_lstm_mrds, color='#0072B2', width=0.6, label='CSBiLSTM - MRDS Wins')
    plt.bar(x, draws, bottom=bi_lstm_mrds + cs_bi_lstm_mrds, color='gray', width=0.6, label='Draws')

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)

    label_mapping = {'downtrend': 'Down-trend',
                     'upshift': 'Up-shift',
                     'downshift': 'Down-shift',
                     'systematic': 'Systematic',
                     'stratification': 'Stratification',
                     'uptrend': 'Up-trend',
                     'cyclic': 'Cyclic'}

    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.set_xticklabels([label_mapping[pattern] for pattern in patterns], rotation=45, fontsize=12)
    fig.patch.set_facecolor('none')

    plt.ylabel('Percentage of Wins/Draws', fontsize=14)
    plt.xlabel('Pattern', fontsize=14)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, ncol=3, fontsize='large')

    plt.tight_layout()
    saving_directory = os.path.join(os.getcwd(), 'Redrawn-Heatmaps')
    os.makedirs(saving_directory, exist_ok=True)
    plt.savefig(os.path.join(saving_directory, 'heatmap_comparisons.pdf'), bbox_inches='tight', transparent=True)
    plt.close()


def main():

    subdirectories = ['Strf=T-CS=F-EQ=T', 'Strf=T-CS=T-EQ=T']
    pattern_names = ["downtrend", "upshift", "downshift", "systematic", "stratification", "uptrend", 'cyclic']

    for pattern_name in pattern_names:
        for subdirectory in subdirectories:
            heatmap_redrawer(subdirectory, pattern_name)

    heatmap_analyzer()


if __name__ == '__main__':
    main()
