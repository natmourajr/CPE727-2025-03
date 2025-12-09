"""
Script for analyzing the impact of Closest Point of Approach (CPA) on classification.

This script generates two tests:

1. Impact of the closest point for CPA:
   - Classifier trained on OS_NEAR_CPA_IN data and evaluated on OS_FAR_CPA_IN data.
   - Classifier trained on OS_FAR_CPA_IN data and evaluated on OS_NEAR_CPA_IN data.

2. Impact of records containing CPA:
   - Classifier trained on OS_CPA_IN data and evaluated on OS_CPA_OUT data.
   - Classifier trained on OS_CPA_OUT data and evaluated on OS_CPA_IN data.
"""
import typing
import itertools
import argparse
import os

import pandas as pd
import torch

import iara.records
import iara.ml.models.mlp as iara_mlp
import iara.ml.models.cnn as iara_cnn
import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES

def main(override: bool,
        folds: typing.List[int],
        only_sample: bool,
        training_strategy: iara_trn.ModelTrainingStrategy,
        cpa_test: int):
    """Grid search main function"""

    if cpa_test == 1:
        collections = [iara.records.Collection.OS_NEAR_CPA_IN,
            iara.records.Collection.OS_FAR_CPA_IN
        ]
    elif cpa_test == 2:
        collections = [iara.records.Collection.OS_CPA_IN,
            iara.records.Collection.OS_CPA_OUT
        ]
    else:
        print('Not implemented test')
        return

    exp_str = 'cpa_exp' if not only_sample else 'cpa_exp_sample'

    result_grid = {}

    output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{exp_str}/exp_{cpa_test}"

    classifiers = [iara_default.Classifier.FOREST, iara_default.Classifier.MLP, iara_default.Classifier.CNN]
    # classifiers = [iara_default.Classifier.MLP]

    manager_dict_0 = iara_default.default_mel_managers(
                config_name = f'{str(collections[0])}',
                output_base_dir = output_base_dir,
                classifiers = classifiers,
                collection = iara_default.default_collection(only_sample=only_sample,
                                                             collection=collections[0]),
                data_processor = iara_default.default_iara_mel_audio_processor(),
                training_strategy = training_strategy)

    manager_dict_1 = iara_default.default_mel_managers(
                config_name = f'{str(collections[1])}',
                output_base_dir = output_base_dir,
                classifiers = classifiers,
                collection = iara_default.default_collection(only_sample=only_sample,
                                                             collection=collections[1]),
                data_processor = iara_default.default_iara_mel_audio_processor(),
                training_strategy = training_strategy)

    for classifier, manager in manager_dict_0.items():
        manager.run(folds = folds, override = override)

    for classifier, manager in manager_dict_1.items():
        manager.run(folds = folds, override = override)

    comparison_dir = f'{output_base_dir}/comparison'

    first_grid = iara_metrics.GridCompiler()
    second_grid = iara_metrics.GridCompiler()

    for classifier in classifiers:
        for strategy in [iara_trn.EvalStrategy.BY_AUDIO]:
        # for strategy in iara_trn.EvalStrategy:

            for subset in [iara_trn.Subset.TEST]:
            # for subset in [iara_trn.Subset.TEST, iara_trn.Subset.ALL]:

                comparator = iara_exp.CrossComparator(comparator_eval_dir = comparison_dir,
                                                    manager_a = manager_dict_0[classifier],
                                                    manager_b = manager_dict_1[classifier])

                result_grid[subset, strategy] = comparator.cross_compare(
                                        eval_subset = subset,
                                        eval_strategy = strategy,
                                        folds = folds)

                print(f'########## {subset}, {strategy} ##########')
                print(result_grid[subset, strategy])

                first_grid.add_cv(result_grid[subset, strategy].get_param_by_index(0), result_grid[subset, strategy].get_cv_by_index(0))
                first_grid.add_cv(result_grid[subset, strategy].get_param_by_index(3), result_grid[subset, strategy].get_cv_by_index(3))
                
                second_grid.add_cv(result_grid[subset, strategy].get_param_by_index(1), result_grid[subset, strategy].get_cv_by_index(1))
                second_grid.add_cv(result_grid[subset, strategy].get_param_by_index(2), result_grid[subset, strategy].get_cv_by_index(2))

    # first_grid.export(os.path.join(output_base_dir, 'first.tex'))
    # second_grid.export(os.path.join(output_base_dir, 'second.tex'))


    first_grid = first_grid.to_df()
    second_grid = second_grid.to_df()

    print('########## first grid ##########')
    print(first_grid)
    print('########## second grid ##########')
    print(second_grid)

    first_grid = first_grid.rename(columns={"SP": f'SP {str(collections[0])}', "ACC": f'ACC {str(collections[0])}'})
    second_grid = second_grid.rename(columns={"SP": f'SP {str(collections[1])}', "ACC": f'ACC {str(collections[1])}'})

    # Realizando o merge com base nas colunas Trainer e Trained
    merged_df = pd.merge(first_grid[['Trainer', 'Trained', f'SP {str(collections[0])}', f'ACC {str(collections[0])}']],
                        second_grid[['Trainer', 'Trained', f'SP {str(collections[1])}', f'ACC {str(collections[1])}']],
                        on=['Trainer', 'Trained'], how='inner')

    # Exibindo o resultado
    print('########## final grid ##########')
    print(merged_df)
    merged_df.to_latex(os.path.join(output_base_dir, 'cpa_proximity.tex' if cpa_test == 1 else 'cpa_presence.tex'), index=False)


if __name__ == "__main__":
    strategy_choises = [str(i) for i in iara_trn.ModelTrainingStrategy]

    parser = argparse.ArgumentParser(description='RUN MLP grid search analysis')
    parser.add_argument('--override', action='store_true', default=False,
                        help='Ignore old runs')
    parser.add_argument('--only_sample', action='store_true', default=False,
                        help='Execute only in sample_dataset. For quick training and test.')
    parser.add_argument('--training_strategy', type=str, choices=strategy_choises,
                        default=None, help='Strategy for training the model')
    parser.add_argument('-F', '--fold', type=str, default=None,
                    help='Specify folds to be executed. Example: 0,4-7')
    test_choices=[1, 2]
    parser.add_argument('--cpa_test', type=int, choices=test_choices, default=None,
                        help='Choose test option\
                            [1. Impact of the closest point for CPA,\
                            2. Impact of records containing CPA]')

    args = parser.parse_args()

    folds_to_execute = iara.utils.str_to_list(args.fold, list(range(10)))

    strategies = []
    if args.training_strategy is not None:
        index = strategy_choises.index(args.training_strategy)
        strategies.append(iara_trn.ModelTrainingStrategy(index))
    else:
        strategies = iara_trn.ModelTrainingStrategy

    for n_test in test_choices if args.cpa_test is None else [args.cpa_test]:

        for strategy in strategies:

            index = strategy_choises.index(args.training_strategy)
            main(override = args.override,
                folds = folds_to_execute,
                only_sample = args.only_sample,
                training_strategy = iara_trn.ModelTrainingStrategy(index),
                cpa_test=n_test)
