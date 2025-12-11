#!/bin/bash

# python training_scripts/grid_forest_mel.py --training_strategy multiclass
# python training_scripts/grid_forest_lofar.py --training_strategy multiclass
# python training_scripts/grid_mlp_lofar.py --training_strategy multiclass

python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 1
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 2
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 3
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 4
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 5
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar -g 6
python training_scripts/grid_search.py -c cnn -t multiclass -f lofar --only_eval