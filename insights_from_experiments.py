
import os
from turtle import width

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
pd.set_option('display.max_columns', None)


def get_combined_method(row):
    method = row['method']
    cost_sensitivity = row['cost_sensitivity']

    if method == 'MRDS' and cost_sensitivity:
        return 'MRDS-CS'
    elif method == 'MRDS' and not cost_sensitivity:
        return 'MRDS-CI'
    elif method == 'conventional' and cost_sensitivity:
        return 'Conv-CS'
    elif method == 'conventional' and not cost_sensitivity:
        return 'Conv-CI'
    else:
        raise ValueError("Error in the combined methods")


df = pd.read_csv(os.path.join(os.getcwd(), "Synthetic Comparison", "synthetic_comparisons.csv"))
df['Methods'] = df.apply(get_combined_method, axis=1)
df.rename(columns={'Pattern-Name': 'Pattern Name', 'time_window': 'Time Window',
                   'abnormality_value': 'Abnormality Value'}, inplace=True)

replacement_dict = {
    'cyclic': 'Cyclic',
    'systematic': 'Systematic',
    'stratification': 'Stratification',
    'downtrend': 'Down-trend',
    'uptrend': 'Up-trend',
    'upshift': "Up-shift",
    'downshift': 'Down-shift'

}

df['Pattern Name'] = df['Pattern Name'].replace(replacement_dict)


def draw_wins_per_time_window(time_window, dataframe):
    df = dataframe[dataframe['Time Window'] == time_window]

    pivot_table = df.pivot_table(index=['Pattern Name', 'Abnormality Value'], columns='Methods',
                                 values='btest_gmean', aggfunc='first').reset_index()

    methods = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']

    pivot_table['Winning Value'] = pivot_table[methods].max(axis=1)

    for column in methods:
        pivot_table[f'{column} Win'] = (pivot_table[column] >= pivot_table['Winning Value']).astype(int)

    win_columns = [method + ' Win' for method in methods]
    total_wins = pivot_table.groupby(['Pattern Name'])[win_columns].sum().reset_index()
    total_wins.columns = total_wins.columns.str.replace(' Win', '')
    melted_data = pd.melt(total_wins, id_vars=['Pattern Name'], var_name='Methods', value_name='Total Wins')

    fig, ax = plt.subplots(figsize=(10, 6))
    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]

    sb.barplot(x='Pattern Name', y='Total Wins', hue='Methods', data=melted_data, palette=custom_palette)
    plt.yticks(range(1, 16), fontsize=16)
    # plt.title('Total Number of Wins per Method and Pattern')
    plt.xticks(rotation=45)
    plt.xlabel('Pattern', fontsize=20)
    plt.ylabel('Total Number of Wins', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=20)

    plt.legend(title='Methods', loc='upper right', fontsize='18', fancybox=True, title_fontsize='20',
               bbox_to_anchor=(1, 1), ncol=2)

    plt.tight_layout()
    plt.savefig(f'.\\Synthetic Comparison\\wins_comparison_T={time_window}.png', format='png')
    plt.close()


def draw_average_btest_gmeans_per_time_window(time_window, dataframe):
    df = dataframe[dataframe['Time Window'] == time_window]
    pivot_table = df.pivot_table(index=['Pattern Name', 'Abnormality Value'], columns='Methods',
                                 values='btest_gmean', aggfunc='first').reset_index()

    # methods = ['Conventional (CI)', 'Conventional (CS)', 'MRDS (CI)', 'MRDS (CS)']
    methods = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']
    group_df = pivot_table.groupby(["Pattern Name"])[methods].mean().reset_index()
    melted_data = group_df.melt(id_vars='Pattern Name', var_name='Methods', value_name='G-means')

    plt.figure(figsize=(10, 6))
    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    sb.barplot(x='Pattern Name', y='G-means', hue='Methods', data=melted_data, palette=custom_palette)
    plt.yticks(np.arange(0, 1.8, 0.1), fontsize=16)
    plt.xticks(rotation=45, fontsize=20)
    plt.xlabel('Pattern', fontsize=20)
    plt.ylabel('Average of G-means', fontsize=20)
    plt.legend(title='Methods', loc='upper right', fontsize='18', fancybox=True, title_fontsize='20',
               bbox_to_anchor=(1, 0.95), ncol=2)
    plt.tight_layout()

    plt.savefig(f'.\\Synthetic Comparison\\average_gmean_comparison_T={time_window}.png', format='png')
    plt.close()


def early_stopping_wins_comparison():
    df = pd.read_csv(os.path.join(os.getcwd(), "Synthetic Comparison", "early_stoppings.csv"))
    df.rename(columns={'Pattern-Name': 'Pattern Name', 'time_window': 'Time Window',
                       'abnormality_value': 'Abnormality Values', 'monitoring_method': 'Monitoring Method',
                       'iteration': 'Iteration', 'kappa': 'Kappa'}, inplace=True)

    df = df[df['Iteration'] == "iteration 1"]
    pivot_table = df.pivot_table(index=['Pattern Name', 'Kappa'], columns='Abnormality Values',
                                 values='btest_gmean', aggfunc='first').reset_index()
    columns_to_compare = pivot_table.columns[2:]
    group_max = pivot_table.groupby("Pattern Name")[columns_to_compare].transform('max')

    wins_columns = []
    for column in columns_to_compare:
        wins_column = str(column) + '_Comparison'
        wins_columns.append(wins_column)
        pivot_table[wins_column] = (pivot_table[column] >= group_max[column]).astype(int)

    pivot_table['Wins'] = pivot_table[wins_columns].sum(axis=1)

    pattern_colors = {'cyclic': '#1f77b4', 'downshift': '#ff7f0e', 'downtrend': '#2ca02c', 'stratification': '#d62728',
                      'systematic': '#9467bd', 'upshift': '#8c564b', 'uptrend': '#e377c2'}

    for pattern in pattern_colors.keys():
        pattern_data = pivot_table[pivot_table["Pattern Name"] == pattern]
        max_wins = pattern_data['Wins'].max()
        custom_palette = ['#4682B4' if wins == max_wins else '#FF6347' for wins in pattern_data['Wins']]

        plt.figure(figsize=(8, 6))
        ax = sb.barplot(x='Kappa', y='Wins', data=pattern_data, palette=custom_palette, dodge=True, linewidth=1.)

        plt.xlabel(r'$\kappa$', fontsize=18)
        plt.ylabel('Total Wins', fontsize=18)

        kappa_values = sorted(pivot_table['Kappa'].unique())
        # custom_labels = ['Loss' if kappa == 0.0 else 'G-mean' if kappa == 60.0 else f"{kappa:.2f}" for kappa in
        #                  kappa_values]
        custom_labels = ['0' if kappa == 0.0 else r'$\infty$' if kappa == 60.0 else f"{kappa:.2f}" for kappa in
                         kappa_values]
        ax.set_xticklabels(custom_labels)
        plt.xticks(rotation=45)

        max_value = pattern_data['Wins'].max()
        ax.axhline(max_value, color='#4682B4', linestyle='--', linewidth=1)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        xtick_labels = ax.get_xticklabels()
        xtick_labels[-1].set_fontsize(30)
        xtick_labels[0].set_fontsize(26)

        plt.tight_layout()
        plt.savefig(f'.\\Synthetic Comparison\\early_stopping_wins_comparison_{pattern}.png', format='png')
        plt.close()

    # all on the single figure
    plt.figure(figsize=(8, 6))
    ax = sb.barplot(x='Kappa', y='Wins', hue='Pattern Name', data=pivot_table, palette=pattern_colors.values(),
                    dodge=True, linewidth=1.5)

    plt.xlabel(r'$\kappa$', fontsize=18)
    plt.ylabel('Total Wins', fontsize=18)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    ax.legend(title='Pattern Name', loc='upper left', fontsize='medium', fancybox=True, title_fontsize='large', ncol=2)

    kappa_values = sorted(pivot_table['Kappa'].unique())
    custom_labels = ['Loss' if kappa == 0.0 else 'G-mean' if kappa == 60.0 else f"{kappa:.2f}"
                     for kappa in kappa_values]
    ax.set_xticklabels(custom_labels)
    plt.xticks(rotation=45)

    for pattern in pivot_table['Pattern Name'].unique():
        pattern_data = pivot_table[pivot_table['Pattern Name'] == pattern]
        max_value = pattern_data['Wins'].max()
        ax.axhline(max_value, color=pattern_colors[pattern], linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(f'.\\Synthetic Comparison\\early_stopping_wins_comparison.png', format='png')
    plt.close()


def geometric_mean(x):
    return np.exp(np.mean(np.log(x)))


def early_stopping_aggregated_gmean_comparison():
    df = pd.read_csv(os.path.join(os.getcwd(), "Synthetic Comparison", "early_stoppings.csv"))
    df.rename(columns={'Pattern-Name': 'Pattern Name', 'time_window': 'Time Window',
                       'abnormality_value': 'Abnormality Values', 'monitoring_method': 'Monitoring Method',
                       'iteration': 'Iteration', 'kappa': 'Kappa'}, inplace=True)

    df = df[df['Iteration'] == "iteration 2"]

    pivot_table = df.pivot_table(index=['Pattern Name', 'Kappa'],
                                 values='btest_gmean', aggfunc=geometric_mean).reset_index()

    for pattern in pivot_table['Pattern Name'].unique():
        pattern_data = pivot_table[pivot_table['Pattern Name'] == pattern]

        plt.figure(figsize=(10, 6))
        ax = sb.lineplot(x='Kappa', y='btest_gmean', data=pattern_data, linewidth=3.0)
        #
        plt.xlabel(r"$\kappa$", fontsize=35)
        plt.ylabel("Average G-means", fontsize=26)
        ax.yaxis.set_major_formatter('{:.3f}'.format)
        plt.yticks(fontsize=26)

        # plt.legend(fontsize=12)
        kappas = np.geomspace(1e-2, 50, 15)
        ax.set_xticks(kappas)
        ax.set_xticklabels([f"{k:.1f}" for k in kappas], rotation=45)

        # custom_kappas = [-4, -1, 1.39, 4.39, 8.06, 14.81, 27.21, 50.0, 60.0]
        custom_kappas = [0.0, 2.2, 5.0, 8.1, 14.8, 27.2, 50.0, 60.0]
        ax.set_xticks(custom_kappas)
        # ax.set_xticklabels(['Loss', '0.71', '2.39', '4.39', '8.06', '14.81', '27.21', '50.00', 'G-mean'], rotation=45)
        ax.set_xticklabels(['0', '2.4', '4.4', '8.1', '14.8', '27.2', '50.0', r'$\infty$'], rotation=45)

        plt.xticks(fontsize=24)

        xtick_labels = ax.get_xticklabels()
        xtick_labels[-1].set_fontsize(30)

        plt.tight_layout()
        plt.savefig(f'.\\Synthetic Comparison\\early_stopping_average_gmeans_{pattern}.png', format='png')
        plt.close()


# arl analysis
def average_time_to_detect_analysis():

    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    columns_to_plot = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']

    # create a custom legend for all figures
    legend_fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    for color, label in zip(custom_palette, columns_to_plot):
        plt.bar(0, 0, color=color, label=label)

    legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot), fontsize='x-large')
    for handle in legend.legendHandles:
        handle.set_height(14)
    legends_saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'Time-To-Detect')
    os.makedirs(legends_saving_directory, exist_ok=True)
    legend_fig.savefig(os.path.join(legends_saving_directory, 'arl_legends.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(legend_fig)

    patterns = ["downtrend", "upshift", "downshift", "systematic", "stratification", "cyclic", "uptrend"]
    # patterns = ["upshift"]
    for pattern in patterns:
        df = pd.read_csv(os.path.join(os.getcwd(), "Early Detection Analysis",
                                      "Detection Times", pattern, f"{pattern}.csv"))

        df.rename(columns={"mixed_CS=T": "MRDS-CS", "mixed_CS=F": "MRDS-CI",
                           "convn_CS=T": "Conv-CS", "convn_CS=F": "Conv-CI"}, inplace=True)

        average_time_to_detect = df.groupby(["Pattern Name", "Time Window", "Imbalanced Ratio",
                                             "Abnormality Value"], as_index=False).mean()

        grouped_atd = average_time_to_detect.groupby(["Pattern Name", "Time Window", "Imbalanced Ratio"],
                                                     as_index=False).mean()

        time_windows = df['Time Window'].unique()
        imb_ratios = df['Imbalanced Ratio'].unique()

        all_figs, axes = plt.subplots(nrows=len(time_windows), ncols=len(imb_ratios),
                                      figsize=(20, 12),
                                      sharex=True, sharey=True)

        plt.ylim(0, 62)

        for i, time_window in enumerate(time_windows):
            for j, imb_ratio in enumerate(imb_ratios):
                data = grouped_atd[(grouped_atd["Pattern Name"] == pattern) &
                                   (grouped_atd["Time Window"] == time_window) &
                                   (grouped_atd["Imbalanced Ratio"] == imb_ratio)]

                sb.barplot(data=data[columns_to_plot], ax=axes[i, j], width=0.5, palette=custom_palette)
                # axes[i, j].set_title(f"T: {time_window}, IR: {imb_ratio}")
                if not j == 0:
                    axes[i, j].tick_params(axis='y', which='both', length=0)

                axes[i, j].tick_params(axis='x', which='both', length=0)
                axes[i, j].set_xticks([])
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_axisbelow(True)

                for p in axes[i, j].patches:
                    axes[i, j].annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                                        fontsize=20)

                axes[i, j].yaxis.set_tick_params(labelsize=20)

        saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'Time-To-Detect')
        os.makedirs(saving_directory, exist_ok=True)
        plt.tight_layout()
        all_figs.savefig(os.path.join(saving_directory, f'arl_{pattern}.png'), dpi=300, bbox_inches='tight')
        print('done')


def count_nan(series):
    size = series.shape[0]
    no_na = series.isna().sum()
    failure_to_detect_percentage = (no_na / size) * 100
    return failure_to_detect_percentage


# Number of detection failures
def average_detection_failure():

    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    columns_to_plot = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']

    # create a custom legend for all figures
    legend_fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    for color, label in zip(custom_palette, columns_to_plot):
        plt.bar(0, 0, color=color, label=label)

    legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot), fontsize='x-large')
    for handle in legend.legendHandles:
        handle.set_height(14)
    legends_saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", "Failure-Stats")
    os.makedirs(legends_saving_directory, exist_ok=True)
    legend_fig.savefig(os.path.join(legends_saving_directory, 'adf_legends.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(legend_fig)

    patterns = ["downtrend", "upshift", "downshift", "systematic", "stratification", "cyclic", "uptrend"]
    # patterns = ["upshift"]
    ylims = {"downtrend": 5, "upshift": 15, "downshift": 18, "systematic": 15,
             "stratification": 57, "cyclic": 38, "uptrend": 5}
    for pattern in patterns:
        df = pd.read_csv(os.path.join(os.getcwd(), "Early Detection Analysis",
                                      "Detection Times", pattern, f"{pattern}.csv"))

        df.rename(columns={"mixed_CS=T": "MRDS-CS", "mixed_CS=F": "MRDS-CI",
                           "convn_CS=T": "Conv-CS", "convn_CS=F": "Conv-CI"}, inplace=True)

        failure_percentage = df.groupby(["Pattern Name",
                                         "Imbalanced Ratio",
                                         "Time Window",
                                         "Abnormality Value"]).agg({"MRDS-CS": count_nan,
                                                                    "MRDS-CI": count_nan,
                                                                    "Conv-CS": count_nan,
                                                                    "Conv-CI": count_nan}).reset_index()

        grouped_ftd = failure_percentage.groupby(["Pattern Name", "Time Window", "Imbalanced Ratio"],
                                                 as_index=False).mean()

        time_windows = df['Time Window'].unique()
        imb_ratios = df['Imbalanced Ratio'].unique()

        all_figs, axes = plt.subplots(nrows=len(time_windows), ncols=len(imb_ratios),
                                      figsize=(20, 12),
                                      sharex=True, sharey=True)

        plt.ylim(0, ylims[pattern])

        for i, time_window in enumerate(time_windows):
            for j, imb_ratio in enumerate(imb_ratios):
                data = grouped_ftd[(grouped_ftd["Pattern Name"] == pattern) &
                                   (grouped_ftd["Time Window"] == time_window) &
                                   (grouped_ftd["Imbalanced Ratio"] == imb_ratio)]

                sb.barplot(data=data[columns_to_plot], ax=axes[i, j], width=0.5, palette=custom_palette)
                # axes[i, j].set_title(f"T: {time_window}, IR: {imb_ratio}")
                if not j == 0:
                    axes[i, j].tick_params(axis='y', which='both', length=0, labelsize=30)

                axes[i, j].tick_params(axis='x', which='both', length=0, labelsize=30)
                axes[i, j].set_xticks([])
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_axisbelow(True)

                for p in axes[i, j].patches:
                    axes[i, j].annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                                        fontsize=20)

                axes[i, j].yaxis.set_tick_params(labelsize=18)

        saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", "Failure-Stats")
        os.makedirs(saving_directory, exist_ok=True)
        plt.tight_layout()
        all_figs.savefig(os.path.join(saving_directory, f'ftd_{pattern}.png'), dpi=300, bbox_inches='tight')
        failure_percentage.to_csv(os.path.join(saving_directory, f'ftd_{pattern}.csv'), index=False)

        print('done')


# analyzing the hardness of problems
def abnormality_value_analyzer():

    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    columns_to_plot = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']
    line_styles = ['-.', ':', '--', '-']
    markers = ["D", "s", "^", "o"]

    # create a custom legend for all figures
    legend_fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    for color, label, linestyle, marker in zip(custom_palette, columns_to_plot, line_styles, markers):
        plt.plot([], [], color=color, label=label, linestyle=linestyle, marker=marker, linewidth=4.5)

    legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot), fontsize='xx-large')
    for handle in legend.legendHandles:
        handle.set_linewidth(2.5)
        handle.set_visible(True)

    # for color, label in zip(custom_palette, columns_to_plot):
    #     sb.lineplot([], color=color, label=label)
    #
    # legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot))
    # for handle in legend.legendHandles:
    #     handle.set_height(14)

    legends_saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", "Abnormality-Value")
    os.makedirs(legends_saving_directory, exist_ok=True)
    legend_fig.savefig(os.path.join(legends_saving_directory, 'ava_legends.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(legend_fig)

    patterns = ["downtrend", "upshift", "downshift", "systematic", "stratification", "cyclic", "uptrend"]
    # patterns = ["upshift"]
    # ylims = {"downtrend": 5, "upshift": 15, "downshift": 18, "systematic": 15,
    #          "stratification": 57, "cyclic": 38, "uptrend": 5}
    for pattern in patterns:
        df = pd.read_csv(os.path.join(os.getcwd(), "Early Detection Analysis",
                                      "Detection Times", pattern, f"{pattern}.csv"))
        df.rename(columns={"mixed_CS=T": "MRDS-CS", "mixed_CS=F": "MRDS-CI",
                           "convn_CS=T": "Conv-CS", "convn_CS=F": "Conv-CI"}, inplace=True)

        time_windows = df['Time Window'].unique()
        imb_ratios = df['Imbalanced Ratio'].unique()

        all_figs, axes = plt.subplots(nrows=len(time_windows), ncols=len(imb_ratios),
                                      figsize=(20, 12),
                                      sharex=True, sharey=True)

        # plt.ylim(0, ylims[pattern])

        for i, time_window in enumerate(time_windows):
            for j, imb_ratio in enumerate(imb_ratios):

                data = df[(df["Pattern Name"] == pattern) &
                          (df["Time Window"] == time_window) &
                          (df["Imbalanced Ratio"] == imb_ratio)]

                pivot_table = data.pivot_table(index='Abnormality Value', values=columns_to_plot, aggfunc='mean')

                for col, linestyle, color, marker in zip(columns_to_plot, line_styles, custom_palette, markers):
                    sb.lineplot(data=pivot_table[col], ax=axes[i, j], label=col, linestyle=linestyle, linewidth=2.5,
                                color=color, marker=marker, legend=False)

                # sb.lineplot(data=pivot_table, ax=axes[i, j], palette=custom_palette, legend=False, linewidth=2.5)

                if not j == 0:
                    axes[i, j].tick_params(axis='y', which='both', length=0)

                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_xlabel("")
                axes[i, j].set_ylabel("")
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
        saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'Abnormality-Value')
        os.makedirs(saving_directory, exist_ok=True)
        plt.tight_layout()
        all_figs.savefig(os.path.join(saving_directory, f'ava_{pattern}.png'), dpi=300, bbox_inches='tight')
        print('done')


# tasrid analysis
def tasrid_analysis():

    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    columns_to_plot = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']

    # create a custom legend for all figures
    legend_fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    for color, label in zip(custom_palette, columns_to_plot):
        plt.bar(0, 0, color=color, label=label)

    legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot), fontsize='x-large')
    for handle in legend.legendHandles:
        handle.set_height(14)
    legends_saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'TASRID')
    os.makedirs(legends_saving_directory, exist_ok=True)
    legend_fig.savefig(os.path.join(legends_saving_directory, 'tasrid_legends.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(legend_fig)

    # patterns = ["downtrend", "upshift", "downshift", "systematic", "stratification", "cyclic", "uptrend"]
    patterns = ["upshift"]
    for pattern in patterns:
        df = pd.read_csv(os.path.join(os.getcwd(), "Early Detection Analysis",
                                      "Detection Rates", pattern, f"{pattern}.csv"))

        df.rename(columns={"mixed_CS=T": "MRDS-CS", "mixed_CS=F": "MRDS-CI",
                           "convn_CS=T": "Conv-CS", "convn_CS=F": "Conv-CI"},
                  inplace=True)

        df = df[df['Detection Rates Metrics'] == 'Mean True Alert Streaks from Initial Detection']

        grouped_df = df.groupby(["Pattern Name", "Time Window", "Imbalanced Ratio"], as_index=False).mean()

        time_windows = df['Time Window'].unique()
        imb_ratios = df['Imbalanced Ratio'].unique()

        all_figs, axes = plt.subplots(nrows=len(time_windows), ncols=len(imb_ratios), figsize=(20, 12),
                                      sharex=True, sharey=True)

        plt.ylim(0, 1)

        for i, time_window in enumerate(time_windows):
            for j, imb_ratio in enumerate(imb_ratios):
                data = grouped_df[(grouped_df["Pattern Name"] == pattern) &
                                  (grouped_df["Time Window"] == time_window) &
                                  (grouped_df["Imbalanced Ratio"] == imb_ratio)]

                sb.barplot(data=data[columns_to_plot], ax=axes[i, j], width=0.5, palette=custom_palette)
                # axes[i, j].set_title(f"T: {time_window}, IR: {imb_ratio}")
                if not j == 0:
                    axes[i, j].tick_params(axis='y', which='both', length=0)

                axes[i, j].tick_params(axis='x', which='both', length=0)
                axes[i, j].set_xticks([])
                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_axisbelow(True)

                for p in axes[i, j].patches:
                    axes[i, j].annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='center', xytext=(0, 10), textcoords='offset points',
                                        fontsize=20)

                axes[i, j].yaxis.set_tick_params(labelsize=20)

        saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'TASRID')
        os.makedirs(saving_directory, exist_ok=True)
        plt.tight_layout()
        all_figs.savefig(os.path.join(saving_directory, f'tasrid_{pattern}.png'), dpi=300, bbox_inches='tight')
        print('done')


# analyzing the hardness of problems for TASRID
def tasrid_abnormality_value_analyzer():

    custom_palette = ["#9370DB", "#FFA07A", "#FF6347", "#4682B4"]
    columns_to_plot = ['Conv-CI', 'Conv-CS', 'MRDS-CI', 'MRDS-CS']
    line_styles = ['-.', ':', '--', '-']
    markers = ["D", "s", "^", "o"]

    # create a custom legend for all figures
    legend_fig, ax = plt.subplots(figsize=(10, 1))
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    for color, label, linestyle, marker in zip(custom_palette, columns_to_plot, line_styles, markers):
        plt.plot([], [], color=color, label=label, linestyle=linestyle, marker=marker, linewidth=4.5)

    legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot), fontsize='xx-large')
    for handle in legend.legendHandles:
        handle.set_linewidth(2.5)
        handle.set_visible(True)

    # for color, label in zip(custom_palette, columns_to_plot):
    #     sb.lineplot([], color=color, label=label)
    #
    # legend = plt.legend(loc='center', frameon=False, ncol=len(columns_to_plot))
    # for handle in legend.legendHandles:
    #     handle.set_height(14)

    legends_saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", "TASRID")
    os.makedirs(legends_saving_directory, exist_ok=True)
    legend_fig.savefig(os.path.join(legends_saving_directory, 'tasrid1_legends.png'), dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(legend_fig)

    patterns = ["downtrend", "upshift", "downshift", "systematic", "stratification", "cyclic", "uptrend"]
    # patterns = ["upshift"]
    # ylims = {"downtrend": 5, "upshift": 15, "downshift": 18, "systematic": 15,
    #          "stratification": 57, "cyclic": 38, "uptrend": 5}
    for pattern in patterns:
        df = pd.read_csv(os.path.join(os.getcwd(), "Early Detection Analysis", "Detection Rates", pattern,
                                      f"{pattern}.csv"))
        df.rename(columns={"mixed_CS=T": "MRDS-CS", "mixed_CS=F": "MRDS-CI",
                           "convn_CS=T": "Conv-CS", "convn_CS=F": "Conv-CI"}, inplace=True)

        df = df[df['Detection Rates Metrics'] == 'Mean True Alert Streaks from Initial Detection']

        time_windows = df['Time Window'].unique()
        imb_ratios = df['Imbalanced Ratio'].unique()

        all_figs, axes = plt.subplots(nrows=len(time_windows), ncols=len(imb_ratios), figsize=(20, 12),
                                      sharex=True, sharey=True)

        # plt.ylim(0, ylims[pattern])

        for i, time_window in enumerate(time_windows):
            for j, imb_ratio in enumerate(imb_ratios):

                data = df[(df["Pattern Name"] == pattern) &
                          (df["Time Window"] == time_window) &
                          (df["Imbalanced Ratio"] == imb_ratio)]

                pivot_table = data.pivot_table(index='Abnormality Value', values=columns_to_plot, aggfunc='mean')

                for col, linestyle, color, marker in zip(columns_to_plot, line_styles, custom_palette, markers):
                    sb.lineplot(data=pivot_table[col], ax=axes[i, j], label=col, linestyle=linestyle, linewidth=2.5,
                                color=color, marker=marker, legend=False)

                # sb.lineplot(data=pivot_table, ax=axes[i, j], palette=custom_palette, legend=False, linewidth=2.5)

                if not j == 0:
                    axes[i, j].tick_params(axis='y', which='both', length=0)

                axes[i, j].grid(True, alpha=0.3)
                axes[i, j].set_xlabel("")
                axes[i, j].set_ylabel("")
                axes[i, j].yaxis.set_tick_params(labelsize=18)
                axes[i, j].xaxis.set_tick_params(labelsize=18)
        saving_directory = os.path.join(os.getcwd(), "Early Detection Analysis", "ARL", 'TASRID')
        os.makedirs(saving_directory, exist_ok=True)
        plt.tight_layout()
        all_figs.savefig(os.path.join(saving_directory, f'tasrid_per_instances_{pattern}.png'),
                         dpi=300, bbox_inches='tight')
        print('done')


def best_selected_models_performance_comparisons():
    data = {'Pattern Name': ['Up-trend', 'Up-trend',
                             'Down-trend', 'Down-trend',
                             'Up-shift', 'Up-shift',
                             'Down-shift', 'Down-shift',
                             'Systematic', 'Systematic',
                             'Cyclic', 'Cyclic',
                             'Stratification', 'Stratification'],
            'Method': ['BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM',
                       'BiLSTM', 'CS-BiLSTM'],
            'Average G-mean': [0.166, 0.892,
                               0.497, 0.860,
                               0.0, 0.615,
                               0.508, 0.791,
                               0.0, 0.682,
                               0.0, 0.642,
                               0.0, 0.598]}

    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 5))  # Adjust the figure size as needed
    ax = sb.barplot(data=df, x='Pattern Name', y='Average G-mean', hue='Method', palette=['#A6CEE3', '#0072B2'],
                    saturation=0.75, dodge=True, width=0.6)

    # ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0, which='major')

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.xlabel('Pattern', fontsize=14)
    plt.ylabel('Average G-mean', fontsize=14)

    # Show the plot
    plt.xticks(rotation=45, ha="right", fontsize=12) # Rotate x-axis labels for better readability
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend(title='Method', loc='upper right')
    # plt.show()
    saving_directory = 'D:\\PhD Degree\\Research Projects\\LSTM for CCPR'
    plt.savefig(os.path.join(saving_directory, 'models_scores.pdf'), bbox_inches='tight')





# draw_wins_per_time_window(50, df)
# draw_wins_per_time_window(100, df)
# #
# draw_average_btest_gmeans_per_time_window(50, df)
# draw_average_btest_gmeans_per_time_window(100, df)
#
# early_stopping_wins_comparison()
# early_stopping_aggregated_gmean_comparison()


# average_time_to_detect_analysis()
# average_detection_failure()
# abnormality_value_analyzer()

# tasrid_analysis()
# tasrid_abnormality_value_analyzer()


best_selected_models_performance_comparisons()


