import os
import datetime
import math
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import click
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score

from mutils.definitions import MUTILS_DATA_DIR
from ppiformer.definitions import PPIFORMER_WEIGHTS_DIR, PPIFORMER_ROOT_DIR


IMPUTE_VAL = 0.6918341792868649


def precision_at_k(ranks, classes, k):
    # claess are bool
    df = pd.DataFrame({'ranks': ranks, 'classes': classes})
    df = df.nsmallest(k, 'ranks')
    return df['classes'].mean()


def run_test(params: dict):
    # Evaluate a single model on a test set
    command = f"""
    WANDB_START_METHOD="thread" HYDRA_FULL_ERROR=1 python {PPIFORMER_ROOT_DIR}/scripts/run.py \
        test=true \
        experiment=ddg_regression \
        run_name=\\\'{params['checkpoint_path']}\\\' \
        project_name=DDG_REGRESSION_TEST \
        model.test_csv_path=\\\'{params['test_csv_path']}\\\' \
        model.checkpoint_path=\\\'{params['checkpoint_path']}\\\' \
        val_dataloader.fresh={params['fresh']} \
        val_dataloader.dataset=\\\'{params['dataset_val']}\\\' \
        val_dataloader.dataset_max_workers=8 \
        val_dataloader.batch_size=1 \
        val_dataloader._pretransform.ddg_label.df_path={params['df_val']} \
        model.kind={params['model_kind']} \
        trainer.devices=1 \
        trainer.accelerator={params['accelerator']}
    """
    return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)


def aggregate_csv(test_csv_paths: list[Path], out_path: Path, df_ref: Path):
    # Aggregate
    test_csv_paths = list(map(str, test_csv_paths))
    dfs = [pd.read_csv(path) for path in test_csv_paths]
    df = pd.concat(dfs)
    df.columns = ['complex', 'mutstr', 'ddG', 'ddG_pred']  # Pdb, Mutation(s)_cleaned, ddG, ddG_pred
    df['ddG'] = df['ddG'].round(7)
    df = df.groupby(['complex', 'mutstr', 'ddG'], dropna=False).mean().reset_index()

    # Impute predictions if absent
    if df_ref is not None:
        df_imputed = []
        df_ref = pd.read_csv(df_ref)
        df_ref['ddG'] = df_ref['ddG'].round(7)
        
        for _, row in df_ref.iterrows():
            if df['complex'].nunique() == 1:
                df_subset = df[(df['mutstr'] == row['mutstr'])]
            else:
                df_subset = df[(df['complex'] == row['complex']) & (df['mutstr'] == row['mutstr'])]
            if len(df_subset) == 0:
                df_imputed.append([row['complex'], row['mutstr'], row['ddG'], IMPUTE_VAL, True])
            else:
                df_imputed.append([row['complex'], row['mutstr'], row['ddG'], df_subset.iloc[0]['ddG_pred'], False])
        df = pd.DataFrame(df_imputed, columns=['complex', 'mutstr', 'ddG', 'ddG_pred', 'imputed'])

    # Add rank values
    df['rank'] = df['ddG_pred'].rank() / len(df)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def test_skempi(pred_dir: Path):
    # Read test 'Protein 1' -> PDB codes mapping
    with open(MUTILS_DATA_DIR / 'SKEMPI2/test_protein1_to_pdbs.json') as file:
        p1_to_pdbs = json.load(file)

    # Calculate per-PPI performance for all methods
    all_dfs_ppi = []
    for p1, pdbs in p1_to_pdbs.items():
        df_ppi = []
        for path in pred_dir.glob('*.csv'):
            name = path.stem

            # Read PPI df for a method
            df = pd.read_csv(path)
            df = df[df['complex'].apply(lambda c: c in pdbs)]
            df = df.fillna(IMPUTE_VAL)
            df['Method'] = name
            df['Protein 1'] = p1

            # Add metrics
            pred = df['ddG_pred'] < 0
            real = df['ddG'] < 0
            metrics =  {
                'Method': name,
                'Spearman': df['ddG'].corr(df['ddG_pred'], method='spearman'),
                'Pearson': df['ddG'].corr(df['ddG_pred'], method='pearson'),
                'Precision': precision_score(real, pred, zero_division=0),
                'Recall': recall_score(real, pred, zero_division=0),
                'ROC AUC': roc_auc_score(real, -df['ddG_pred']) if len(df) and real.nunique() > 1 else np.nan,
                'PR AUC': average_precision_score(real, -df['ddG_pred']) if len(df) and real.nunique() > 1 else np.nan,
                'MAE': (df['ddG'] - df['ddG_pred']).abs().mean(),
                'RMSE': math.sqrt((df['ddG'] - df['ddG_pred']).pow(2).mean())
            }
            if metrics is not None:
                df_ppi.append(metrics)
        
        df_ppi = pd.DataFrame(df_ppi).set_index('Method')
        print('Protein 1:', p1)
        print((df_ppi[['Spearman', 'Precision', 'Recall']]), end='\n\n')
        all_dfs_ppi.append(df_ppi)

    # Calculate overall performance
    print('Overall')
    print(pd.concat(all_dfs_ppi).round(2).reset_index().groupby(by='Method').mean())


def test_shan2022(pred_dir: Path):
    stabilizing_muts = ['TH31W', 'AH53F', 'NH57L', 'RH103M', 'LH104F']

    # Calculate performance of each method
    df_overall = []
    df_test = []
    for path in pred_dir.glob('*.csv'):
        df = pd.read_csv(path)
        if 'rank' not in df.columns:
            df['rank'] = df['ddG_pred'].rank() / len(df)
        metrics = {}
        metrics['name'] = path.stem

        for k in [1, 25, 49]:
            metrics[f'P@{k}'] = precision_at_k(df['rank'], df['ddG'] < 0, k)
        metrics['Mean rank'] = df[df['mutstr'].isin(stabilizing_muts)]['rank'].mean()

        dct = df[df['mutstr'].isin(stabilizing_muts)][['mutstr', 'rank']].set_index('mutstr').T.to_dict()
        dct = df[df['mutstr'].isin(stabilizing_muts)][['mutstr', 'rank']].set_index('mutstr').T.to_dict()
        dct = {k: v['rank'] for k, v in dct.items()}
        dct['Method'] = path.stem
        df_overall.append(dct)
        df_test.append(metrics)

    # Calculate overall performance
    df_overall = pd.DataFrame(df_overall)
    print((100*df_overall).round(2))
    df_test = pd.DataFrame(df_test).set_index('name')
    print((100*df_test).round(3))


@click.command()
@click.option('--checkpoints', type=Path, default=PPIFORMER_WEIGHTS_DIR / 'ddg_regression')
@click.option('--accelerator', default='cpu')
@click.option('--fresh', default=False, help='Pre-process data from scratch if not present.')
def main(checkpoints, accelerator, fresh):
    # Set testing params
    base_params = {
        'accelerator': accelerator,
        'fresh': fresh,
        'model_kind': 'masked_marginals',
    }
    complexes = False

    # Define models checkpoints to test
    checkpoint_paths = list(checkpoints.glob('*.ckpt'))
    assert len(checkpoint_paths) == 3

    # Test on SKEMPI2
    print(40*'=' + 'SKEMPI test' + 40*'=')
    pred_dir = MUTILS_DATA_DIR / 'SKEMPI2/predictions_test'
    params = copy.deepcopy(base_params)
    params['df_val'] = "null"  # SKEMPI v2.0 is used for ddG labels by default
    test_csv_paths = []
    for checkpoint_path in tqdm(checkpoint_paths, desc='Pre-processing data and making predictions'):
        params['checkpoint_path'] = checkpoint_path
        params['dataset_val'] = 'skempi2_complexes_V3_split,test' if complexes else 'skempi2_V3_split,test'
        params['test_csv_path'] = pred_dir / f'raw/ppiformer_{checkpoint_path.stem}.csv'
        test_csv_paths.append(params['test_csv_path'])
        run_test(params)
    aggregate_csv(
        test_csv_paths,
        out_path=pred_dir / f'ppiformer.csv',
        df_ref=pred_dir / 'results_RDE.csv'
    )
    test_skempi(pred_dir)

    # Test on Shan2022 SARS-CoV-2
    print(40*'=' + 'SARS-CoV-2 test' + 40*'=')
    pred_dir = MUTILS_DATA_DIR / '7FAE/predictions_test'
    params = copy.deepcopy(base_params)
    params['df_val'] = f"\\\'{MUTILS_DATA_DIR}/7FAE/shan2022_covid_SKEMPI_format.csv\\\'"
    test_csv_paths = []
    for checkpoint_path in tqdm(checkpoint_paths, desc='Pre-processing data and making predictions'):
        params['checkpoint_path'] = checkpoint_path
        params['dataset_val'] = 'shan2022_covid_complex,whole' if complexes else 'shan2022_covid,whole'
        params['test_csv_path'] = pred_dir / f'raw/ppiformer_{checkpoint_path.stem}.csv'
        test_csv_paths.append(params['test_csv_path'])
        run_test(params)
    aggregate_csv(
        test_csv_paths,
        out_path=pred_dir / f'ppiformer.csv',
        df_ref=pred_dir / 'results_RDE.csv'
    )
    test_shan2022(pred_dir)


if __name__ == '__main__':
    main()
