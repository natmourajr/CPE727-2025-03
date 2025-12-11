import os
import enum
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import iara.ml.models.trainer as iara_trn
import iara.ml.metrics as iara_metrics
import iara.default

class Feature(enum.Enum):
    MEL = 0
    MEL_GRID = 1
    LOFAR = 2

    def to_str(self) -> str:
        return str(self.name).rsplit(".", maxsplit=1)[-1].lower()

subset = iara_trn.Subset.TEST
feature = Feature.MEL
model_id = {
    iara.default.Classifier.FOREST: '0[0]_1[2]',
    iara.default.Classifier.MLP: '0[1]_1[9]_2[1]_3[0]_4[1]_5[0]_6[1]_7[1]_8[0]',
    iara.default.Classifier.CNN: '0[0]_1[7]_2[2]_3[1]_4[0]_5[2]_6[0]_7[1]_8[2]_9[2]_10[2]_11[0]_12[0]_13[1]_14[3]_15[0]',
}
percent_list = list(np.arange(25, 100, 1))
folds = range(10)

grid = iara_metrics.GridCompiler()
qtd = {}

for classifier in iara.default.Classifier:

    qtd[classifier] = {
        'x': percent_list,
        'y': np.zeros((len(percent_list), len(folds)))
    }
        
    for i_fold in folds:

        model_base_dir = f"./results/trainings/grid_search/{classifier}_{feature.to_str()}_multiclass/eval/fold_{i_fold}"

        filename = os.path.join(model_base_dir, f'{model_id[classifier]}_multiclass_{subset}.csv')
        
        if os.path.exists(filename):

            df = pd.read_csv(filename)

            def most_common_value(series):
                counter = collections.Counter(series)
                most_common, count = counter.most_common(1)[0]
                total_count = sum(counter.values())
                percentage = (count / total_count) * 100
                return most_common, percentage
                # return collections.Counter(series).most_common(1)[0][0]

            df = df.groupby('File').agg({
                'Target': most_common_value,
                'Prediction': most_common_value
            }).reset_index()

            df['Target'], _ = zip(*df['Target'])
            df['Prediction_Label'], df['Prediction_Percentage'] = zip(*df['Prediction'])

            df.drop(columns=['Prediction'], inplace=True)

            for min_percent in percent_list:
                filtered_df = df[df['Prediction_Percentage'] > min_percent]

                params = {
                    'classifier': classifier,
                    'min_percent': min_percent,
                }

                grid.add(params=params, i_fold=i_fold, target=filtered_df['Target'], prediction=filtered_df['Prediction_Label'])

                qtd[classifier]['y'][percent_list.index(min_percent), i_fold] = len(filtered_df)


print(grid)


output_dir = './results/confidence_plots/'
os.makedirs(output_dir, exist_ok=True)

metric_list = [iara_metrics.Metric.SP_INDEX,
                iara_metrics.Metric.BALANCED_ACCURACY,
                iara_metrics.Metric.MACRO_F1]
for metric in metric_list:

    x_values = {}
    y_means = {}
    y_stds = {}

    index = 0
    for classifier in iara.default.Classifier:

        x_values[classifier] = []
        y_means[classifier] = []
        y_stds[classifier] = []

        for _ in percent_list:
            params = grid.get_param_by_index(index)
            cv = grid.get_cv_by_index(index)
            metric_values = cv.get(metric)

            x_values[classifier].append(params['min_percent'])
            y_means[classifier].append(np.mean(metric_values))
            y_stds[classifier].append(np.std(metric_values))

            index = index + 1


        # grafico individual da metrica
        plt.figure(figsize=(10, 6))
        plt.errorbar(x_values[classifier], y_means[classifier], yerr=y_stds[classifier], fmt='o', capsize=5)
        plt.title(f'Performance of {metric} for {classifier}')
        plt.xlabel('Min Percent')
        plt.ylabel(f'Mean {metric}')
        plt.grid(True)

        plt.savefig(f'{output_dir}{metric}_{classifier}.png')
        plt.close()


        # print(qtd[classifier]['x'])
        # print(np.mean(qtd[classifier]['y'], axis = 1))
        # grafico duplo da metrica
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.errorbar(x_values[classifier], y_means[classifier], yerr=y_stds[classifier],  fmt='o', capsize=5, label=f'{metric} Mean')
        ax1.set_xlabel('Min Percent')
        ax1.set_ylabel(f'Mean {metric}')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(qtd[classifier]['x'], np.mean(qtd[classifier]['y'], axis = 1), 'r-', label='Qtd Dados')
        ax2.set_ylabel('Qtd Dados', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        fig.suptitle(f'Performance of {metric} for {classifier}')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
        plt.savefig(f'{output_dir}{metric}_{classifier}_len.png')
        plt.close()



    # grafico compilado
    plt.figure(figsize=(10, 6))
    for classifier in iara.default.Classifier:
        plt.errorbar(x_values[classifier], y_means[classifier], yerr=y_stds[classifier], fmt='o', capsize=5, label=classifier)
    plt.title(f'Performance of {metric} for {classifier}')
    plt.xlabel('Min Percent')
    plt.ylabel(f'Mean {metric}')
    plt.legend(title='Classifier')
    plt.grid(True)

    plt.savefig(f'{output_dir}{metric}.png')
    plt.close()

grid.export(os.path.join(output_dir,'grid.tex'))
grid.export(os.path.join(output_dir,'grid.csv'))


dataframes = []
for metric in metric_list:
    for classifier in iara.default.Classifier:
        x = qtd[classifier]['x']
        y = np.mean(qtd[classifier]['y'], axis=1)
        
        df = pd.DataFrame({'x': x, 'y': y})
        
        df['metric'] = metric
        df['classifier'] = classifier
        
        dataframes.append(df)

# Concatenando todos os DataFrames temporários em um único DataFrame
final_df = pd.concat(dataframes, ignore_index=True)

print(final_df)
final_df.to_csv(os.path.join(output_dir,'len_grid.csv'))
