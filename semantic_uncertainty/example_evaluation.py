# Fill in the wandb_id assigned to your demo run!
wandb_id = 'qq4wfsfc' # 'bydbd6hl' # 'YOUR_ID'
if wandb_id == 'YOUR_ID':
    raise ValueError('Need to provide wandb_id of demo run!')

import os
import json
import wandb
import pandas as pd
from matplotlib import pyplot as plt

# Helper Functions

def restore_file(wandb_id, filename='wandb-summary.json'):
    files_dir = 'notebooks/restored_files'    
    os.system(f'mkdir -p {files_dir}')

    api = wandb.Api()
    run = api.run(f'semantic_uncertainty/{wandb_id}')

    path = f'{files_dir}/{filename}'
    os.system(f'rm -rf {path}')
    run.file(filename).download(root=files_dir, replace=True, exist_ok=False)
    with open(path, 'r') as f:
        out = json.load(f)
    return out

def get_uncertainty_df(metrics):
    data = []
    for method in metrics['uncertainty']:
        for metric in metrics['uncertainty'][method]:
            mean = metrics['uncertainty'][method][metric]['mean']
            data.append([method, metric, mean])
    df = pd.DataFrame(data, columns=['method', 'metric', 'means'])
    main_methods = ['semantic_entropy', 'recalibrated_semantic_entropy', 'cluster_assignment_entropy', 'regular_entropy', 'p_false', 'p_ik']
    df = df.set_index('method').loc[main_methods].reset_index()
    main_names = ['Semantic entropy', 'Recalibrated Semantic Entropy', 'Discrete Semantic Entropy', 'Naive Entropy', 'p(True)', 'Embedding Regression']
    conversion = dict(zip(main_methods, main_names))
    df['method'] = df.method.map(lambda x: conversion[x])
    return df

def main():
    results = restore_file(wandb_id)
    unc_df = get_uncertainty_df(results)

    # Create plots directory if it doesn't exist
    os.makedirs('./plots', exist_ok=True)

    metric = 'AUROC'
    unc_df.set_index('metric').loc[metric].plot.bar(x='method', y='means')
    plt.gca().set_ylabel(metric)
    plt.gca().grid(axis='y')
    plt.gca().set_ylim(0.6, 0.8)

    # Save the plot
    plt.savefig(f'./plots/run_{wandb_id}_uncertainty_{metric.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()