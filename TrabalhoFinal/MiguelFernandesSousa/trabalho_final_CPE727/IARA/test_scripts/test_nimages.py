import itertools
import pandas as pd

import iara.ml.experiment as iara_exp
import iara.ml.dataset as iara_dataset
import iara.records

import iara.default as iara_default
from iara.default import DEFAULT_DIRECTORIES

# params = {
#     'Processor': [iara_default.default_iara_lofar_audio_processor(),
#                   iara_default.default_iara_mel_audio_processor()],
#     'Input': [iara_default.default_image_input(),
#               iara_default.default_window_input()]
# }
params = {
    'Processor': [iara_default.default_iara_mel_audio_processor()],
    'Input': [iara_default.default_window_input()]
}

custom_collection = iara_default.default_collection(collection=iara.records.Collection.C)
output_base_dir = f"{DEFAULT_DIRECTORIES.training_dir}/test_image_loader"

combinations = list(itertools.product(*params.values()))
for combination in combinations:
    param_pack = dict(zip(params.keys(), combination))

    print("#################################################################")
    print(f"--- Processor[{param_pack['Processor'].analysis}] --- Input[{param_pack['Input']}]")

    config = iara_exp.Config(
                    name = 'image_loader',
                    dataset = custom_collection,
                    dataset_processor = param_pack['Processor'],
                    input_type = param_pack['Input'],
                    output_base_dir = output_base_dir,
                    exclusive_ships_on_test=True,
                    test_ratio = 0.25)


    id_list = config.split_datasets()

    df = config.dataset.to_compiled_df()
    df = df.rename(columns={'Qty': 'Total'})

    for i_fold, (trn_set, val_set, test_set) in enumerate(id_list):

        df_trn = config.dataset.to_compiled_df(trn_set)
        df_val = config.dataset.to_compiled_df(val_set)
        df_test = config.dataset.to_compiled_df(test_set)

        df_trn = df_trn.rename(columns={'Qty': f'Trn_{i_fold}'})
        df_val = df_val.rename(columns={'Qty': f'Val_{i_fold}'})
        df_test = df_test.rename(columns={'Qty': f'Test_{i_fold}'})

        print(test_set['ID'].to_list())


        # df = pd.merge(df, df_trn, on=config.dataset.target.grouped_column())
        # df = pd.merge(df, df_val, on=config.dataset.target.grouped_column())
        df = pd.merge(df, df_test, on=config.dataset.target.grouped_column())
        # break

    print(f'--- Dataset with {len(id_list)} n_folds ---')
    print(df)


    # df = config.dataset.to_df()
    # exp_loader = config.get_data_loader()

    # trn_set, val_set, test_set = id_list[0]
    # exp_loader.pre_load(trn_set['ID'].to_list())
    # exp_loader.pre_load(val_set['ID'].to_list())
    # exp_loader.pre_load(test_set['ID'].to_list())

    # print('--- Details ---')
    # trn_dataset = iara_dataset.AudioDataset(exp_loader,
    #                                         config.input_type,
    #                                         trn_set['ID'].to_list())
    # print('trn_dataset: ', trn_dataset)
    # val_dataset = iara_dataset.AudioDataset(exp_loader,
    #                                         config.input_type,
    #                                         val_set['ID'].to_list())
    # print('val_dataset: ', val_dataset)
    # test_dataset = iara_dataset.AudioDataset(exp_loader,
    #                                         config.input_type,
    #                                         test_set['ID'].to_list())
    # print('test_dataset: ', test_dataset)
