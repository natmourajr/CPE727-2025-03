import os
import pprint

import iara.default
import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

from iara.default import DEFAULT_DIRECTORIES

import grid_search

training_strategy = iara_trn.ModelTrainingStrategy.MULTICLASS
# grid_str = 'shipsear_grid'
grid_str = 'grid_search'
output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/{grid_str}"

result_dict = {}

for eval_strategy in iara_trn.EvalStrategy:

    grid = iara_metrics.GridCompiler()
    best_params_dict = {}

    for classifier in iara.default.Classifier:
        for feature in [grid_search.Feature.MEL, grid_search.Feature.LOFAR]:

            compiled_dir = f'{output_base_dir}/compiled'

            filename = f'{compiled_dir}/{classifier}_{feature}_{training_strategy}_{eval_strategy}.pkl'

            if os.path.exists(filename):
                gc = iara_metrics.GridCompiler.load(filename)
                param, cv = gc.get_best()

                best_params_dict[str(classifier), str(feature)] = param

                grid.add_cv(cv=cv,
                            params={
                                'Feature': feature,
                                'Classifier': classifier
                            })

    print(f"\n############# {eval_strategy} #############")
    pprint.pprint(best_params_dict)
    print(grid)
    grid.export(f'{output_base_dir}/{eval_strategy}.pkl')
    grid.export(f'{output_base_dir}/{eval_strategy}.tex')

