"""
Dataset Info Tables Test Program

This script generates tables describing compiled information.
In the future, this test should lead to an application that generates .tex tables
    to be used in the dataset publication article.
"""
import os
import pandas as pd

import iara.records
from iara.default import DEFAULT_DIRECTORIES


def main(show_sample_dataset = False):
    """Main function for the dataset info tables."""

    os.makedirs(DEFAULT_DIRECTORIES.tables_dir, exist_ok=True)

    os_ship_merged = []
    collection_list = [
        iara.records.Collection.A,
        iara.records.Collection.B,
        iara.records.Collection.C,
        iara.records.Collection.D,
    ]
    for sub in collection_list:
        df = sub.to_df(only_sample=show_sample_dataset)
        part = df.groupby(['SHIPTYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(os_ship_merged, pd.DataFrame):
            os_ship_merged = part
        else:
            os_ship_merged = pd.merge(os_ship_merged, part,
                                      on=['SHIPTYPE','DETAILED TYPE'],how='outer')

    os_ship_merged = os_ship_merged.fillna(0)
    os_ship_merged = os_ship_merged.sort_values(['SHIPTYPE','DETAILED TYPE'])
    os_ship_merged['Total'] = os_ship_merged[os_ship_merged.columns[2:]].sum(axis=1)

    keeped = os_ship_merged[os_ship_merged['Total']>=20]

    filtered = os_ship_merged[os_ship_merged['Total']<20]
    filtered = filtered.groupby('SHIPTYPE').sum()
    filtered['DETAILED TYPE'] = 'Others'
    filtered.reset_index(inplace=True)

    os_ship_detailed_type = pd.concat([keeped, filtered])
    os_ship_detailed_type.sort_values(by='DETAILED TYPE', inplace=True)
    os_ship_detailed_type.sort_values(by='SHIPTYPE', inplace=True)
    os_ship_detailed_type.reset_index(drop=True, inplace=True)
    os_ship_detailed_type.loc['Total'] = os_ship_detailed_type.sum()
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'SHIPTYPE'] = 'Total'
    os_ship_detailed_type.loc[os_ship_detailed_type.index[-1], 'DETAILED TYPE'] = 'Total'
    os_ship_detailed_type[os_ship_detailed_type.columns[2:]] = \
            os_ship_detailed_type[os_ship_detailed_type.columns[2:]].astype(int)
    os_ship_detailed_type.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_ship_detailed_type.tex'), index=False)

    print('------------------------ os_ship_detailed_type ----------------------------------------')
    print(os_ship_detailed_type)


    os_ship_type = os_ship_merged.groupby('SHIPTYPE').sum()
    os_ship_type = os_ship_type.drop('DETAILED TYPE', axis=1)
    os_ship_type.loc['Total'] = os_ship_type.sum()
    os_ship_type[os_ship_type.columns] = \
            os_ship_type[os_ship_type.columns].astype(int)
    os_ship_type.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_ship_type.tex'), index=True)

    print('------------------------ os_ship_type ----------------------------------------')
    print(os_ship_type)


    os_bg = iara.records.Collection.E.to_df(only_sample=show_sample_dataset)

    os_with_rain = os_bg[os_bg['Rain state']=='No rain']
    os_without_rain = os_bg[os_bg['Rain state']!='No rain']

    os_with_rain = os_with_rain.groupby('Sea state').size().reset_index(name='With rain')
    os_without_rain = os_without_rain.groupby('Sea state').size().reset_index(name='Without rain')

    os_sea_state =  pd.merge(os_with_rain, os_without_rain, on=['Sea state'],how='outer')
    os_sea_state = os_sea_state.fillna(0)
    os_sea_state[os_sea_state.columns[1:]] = \
            os_sea_state[os_sea_state.columns[1:]].astype(int)

    os_sea_state.loc['Total'] = os_sea_state.sum()
    os_sea_state.loc['Total', 'Sea state'] = 'Total'
    os_sea_state.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_bg_sea_state.tex'), index=False)

    os_rain_state = os_bg.groupby('Rain state').size().reset_index(name='Qty')

    order_map = {}
    for state in iara.records.Rain:
        order_map[str(state)] = state.value

    os_rain_state['Rain state order'] = os_rain_state['Rain state'].map(
            lambda x: order_map.get(x, float('nan')))
    os_rain_state = os_rain_state.sort_values(by='Rain state order')
    os_rain_state = os_rain_state.drop(columns='Rain state order')

    os_rain_state.loc['Total'] = os_rain_state.sum()
    os_rain_state.loc['Total', 'Rain state'] = 'Total'

    os_rain_state.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'os_bg_rain_state.tex'), index=False)

    print('\n------------------------- os_sea_state -----------------------------------------')
    print(os_sea_state)

    print('\n------------------------- os_rain -----------------------------------------')
    print(os_rain_state)


    glider_ship_merged = []
    collection_list = [
        iara.records.Collection.F,
        iara.records.Collection.G,
    ]
    for sub in collection_list:
        df = sub.to_df(only_sample=show_sample_dataset)
        part = df.groupby(['SHIPTYPE','DETAILED TYPE']).size().reset_index(name=str(sub))

        if not isinstance(glider_ship_merged, pd.DataFrame):
            glider_ship_merged = part
        else:
            glider_ship_merged = pd.merge(glider_ship_merged, part,
                                      on=['SHIPTYPE','DETAILED TYPE'],how='outer')

    glider_ship_merged = glider_ship_merged.fillna(0)
    glider_ship_merged = glider_ship_merged.sort_values(['SHIPTYPE','DETAILED TYPE'])
    glider_ship_merged['Total'] = glider_ship_merged[glider_ship_merged.columns[2:]].sum(axis=1)

    glider_ship_merged.sort_values(by='DETAILED TYPE', inplace=True)
    glider_ship_merged.sort_values(by='SHIPTYPE', inplace=True)
    glider_ship_merged.reset_index(drop=True, inplace=True)
    glider_ship_merged.loc['Total'] = glider_ship_merged.sum()
    glider_ship_merged.loc[glider_ship_merged.index[-1], 'SHIPTYPE'] = 'Total'
    glider_ship_merged.loc[glider_ship_merged.index[-1], 'DETAILED TYPE'] = 'Total'
    glider_ship_merged[glider_ship_merged.columns[2:]] = \
            glider_ship_merged[glider_ship_merged.columns[2:]].astype(int)
    glider_ship_merged.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'glider_ship_merged.tex'), index=False)

    print('------------------------ glider_ship_detailed_type ----------------------------------------')
    print(glider_ship_merged)


    glider_ship_type = glider_ship_merged.groupby('SHIPTYPE').sum()
    glider_ship_type = glider_ship_type.drop('DETAILED TYPE', axis=1)
    glider_ship_type.loc['Total'] = glider_ship_type.sum()
    glider_ship_type[glider_ship_type.columns] = \
            glider_ship_type[glider_ship_type.columns].astype(int)
    glider_ship_type.to_latex(
            os.path.join(DEFAULT_DIRECTORIES.tables_dir, 'glider_ship_type.tex'), index=True)

    print('------------------------ glider_ship_type ----------------------------------------')
    print(glider_ship_type)



if __name__ == "__main__":
    main()
