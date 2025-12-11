import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES


output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/cross_dataset"
comparison_dir = f'{output_base_dir}/comparison'
grids_dir = f'{output_base_dir}/grids'

subset = iara_trn.Subset.ALL
classifier = iara_default.Classifier.CNN

filename = f'{grids_dir}/{classifier}_{subset}.pkl'
grid = iara_metrics.GridCompiler.load(filename)

print(grid)

for i in range(4):
    print('##########', grid.get_param_by_index(i), '##########')
    cv = grid.get_cv_by_index(i)
    cv.print_cm(relative=False)