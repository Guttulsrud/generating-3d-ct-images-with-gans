import pandas as pd
import yaml
from matplotlib import pyplot as plt
import matplotlib as mpl
import shutil

mpl.use('TkAgg')

import os
import glob

main_dir = "../saved_models_segment/5fold"


def create_plot(data, model, metric, metric_name, show=False, alpha=0.001):
    data[metric_name] = data[metric]
    data[f'Smoothed {metric_name}'] = data[metric].ewm(alpha=alpha).mean()

    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)

    ax.plot(data['epoch'], data[metric_name], alpha=0.2, label=metric_name)
    ax.plot(data['epoch'], data[f'Smoothed {metric_name}'], label=f'{metric_name} (EMA)')
    ax.legend()

    plt.savefig(f'../results/segmentation/{experiment}/{model}/plots/png/{metric_name}.png')
    plt.savefig(f'../results/segmentation/{experiment}/{model}/plots/pdf/{metric_name}.pdf')
    plt.savefig(f'../results/segmentation/{experiment}/{model}/plots/svg/{metric_name}.svg')

    if show:
        plt.show()
    plt.close()


for dirpath, dirnames, filenames in os.walk(main_dir):

    if 'segresnet_4' not in dirnames:
        # Skip results that are not 5fold cross validated
        continue

    # Get dataframes for all runs
    all_dataframes = []
    for dirname in dirnames:
        if "segresnet_" in dirname:
            accuracy_file = os.path.join(dirpath, dirname, "model", "accuracy_history.csv")
            data = pd.read_csv(accuracy_file, delimiter='\t')
            all_dataframes.append(data)

    # Get dataframes for all runs
    all_dataframes = []
    for dirname in dirnames:
        if "segresnet_" in dirname:
            accuracy_file = os.path.join(dirpath, dirname, "model", "accuracy_history.csv")
            data = pd.read_csv(accuracy_file, delimiter='\t')
            # Calculate EMA values for the DCS column
            data[f'Smoothed metric'] = data['metric'].ewm(alpha=0.01, adjust=False).mean()
            all_dataframes.append(data)

    # Create a new plot
    fig, ax = plt.subplots()
    ax.grid(color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DCS')
    for i, run_df in enumerate(all_dataframes):
        ax.plot(run_df['epoch'], run_df['Smoothed metric'], label=f'SegResNet {i + 1} (EMA)')
    ax.legend()

    experiment = dirpath.split('../saved_models_segment\\')[-1]

    if experiment != '../saved_models_segment/5fold\RF_testing_5fold':
        continue

    print(experiment)

    if not os.path.exists(f'../results/segmentation/{experiment}'):
        os.makedirs(f'../results/segmentation/{experiment}')

    for x in ['segresnet_0', 'segresnet_1', 'segresnet_2', 'segresnet_3', 'segresnet_4', 'combined']:
        if not os.path.exists(f'../results/segmentation/{experiment}/{x}/plots'):
            os.makedirs(f'../results/segmentation/{experiment}/{x}/plots')

        if not os.path.exists(f'../results/segmentation/{experiment}/{x}/plots/png'):
            os.makedirs(f'../results/segmentation/{experiment}/{x}/plots/png')

        if not os.path.exists(f'../results/segmentation/{experiment}/{x}/plots/pdf'):
            os.makedirs(f'../results/segmentation/{experiment}/{x}/plots/pdf')

        if not os.path.exists(f'../results/segmentation/{experiment}/{x}/data'):
            os.makedirs(f'../results/segmentation/{experiment}/{x}/data')

        if not os.path.exists(f'../results/segmentation/{experiment}/{x}/plots/svg'):
            os.makedirs(f'../results/segmentation/{experiment}/{x}/plots/svg')

    plt.savefig(f'../results/segmentation/{experiment}/combined/plots/png/all_dcs.png')
    plt.savefig(f'../results/segmentation/{experiment}/combined/plots/pdf/all_dcs.pdf')
    plt.savefig(f'../results/segmentation/{experiment}/combined/plots/svg/all_dcs.svg')
    plt.close()
    dataframes = []
    for dirname in dirnames:

        if "segresnet_" in dirname:
            segresnet_dir = os.path.join(dirpath, dirname)
            model_dir = os.path.join(segresnet_dir, "model")
            progress_file = os.path.join(model_dir, "progress.yaml")
            accuracy_file = os.path.join(model_dir, "accuracy_history.csv")

            shutil.copy(progress_file, f'../results/segmentation/{experiment}/{dirname}/data/progress.yaml')
            shutil.copy(accuracy_file, f'../results/segmentation/{experiment}/{dirname}/data/accuracy_history.csv')

            data = pd.read_csv(accuracy_file, delimiter='\t')

            create_plot(data, dirname, 'metric', 'DCS', show=False, alpha=0.01)
            create_plot(data, dirname, 'loss', 'Loss', show=False, alpha=0.01)
            dataframes.append(data)
            df = pd.DataFrame([{'max_dcs': data['metric'].max(), 'min_loss': data['loss'].min()}])
            pd.DataFrame(df).to_csv(f'../results/segmentation/{experiment}/{dirname}/data/max_dcs_min_loss.csv',
                                    index=False)

    combined_df = pd.concat(dataframes, axis=0)
    mean_metric = combined_df.groupby('epoch')['metric'].mean().reset_index()
    mean_loss = combined_df.groupby('epoch')['loss'].mean().reset_index()
    mean_metric = mean_metric.rename(columns={'metric': 'Average DCS'})
    mean_loss = mean_loss.rename(columns={'loss': 'Average Loss'})
    mean_metric.to_csv(f'../results/segmentation/{experiment}/combined/data/mean_metric.csv', index=False)
    mean_loss.to_csv(f'../results/segmentation/{experiment}/combined/data/mean_loss.csv', index=False)
    create_plot(mean_metric, 'Combined', 'Average DCS', 'DCS', show=False, alpha=0.01)
    create_plot(mean_loss, 'Combined', 'Average Loss', 'Loss', show=False, alpha=0.01)

    df = pd.DataFrame([{'max_dcs': mean_metric['Average DCS'].max(), 'min_loss': mean_loss['Average Loss'].min()}])
    pd.DataFrame(df).to_csv(f'../results/segmentation/{experiment}/combined/data/max_dcs_min_loss.csv',
                            index=False)

    result_data = []
    for dirname in dirnames + ['combined']:
        if "segresnet_" in dirname or 'combined' in dirname:
            max_dcs_min_loss_file = os.path.join(f'../results/segmentation/{experiment}', dirname, 'data',
                                                 'max_dcs_min_loss.csv')
            if os.path.exists(max_dcs_min_loss_file):
                df = pd.read_csv(max_dcs_min_loss_file)
                max_dcs = df['max_dcs'].iloc[0]
                min_loss = df['min_loss'].iloc[0]
                result_data.append({'model': dirname, 'max dcs': max_dcs, 'min loss': min_loss})

    result_df = pd.DataFrame(result_data)
    result_df.columns = ['Model', 'Max DCS', 'Min Loss']
    # result_df = result_df.iloc[1:]
    result_df = result_df.sort_values(by=['Max DCS'], ascending=False)
    result_df.to_csv(f'../results/segmentation/{experiment}/combined/data/results.csv', index=False)
