import pandas as pd
import os
import glob
import math
import json


def get_df(item_path, scores_path):
    items_list = []
    for file_i in glob.iglob((item_path + '/*.tsv')):
        item_df = pd.read_csv(file_i)
        items_list.append(item_df)
    scores_list = []
    for file_s in glob.iglob((scores_path + '/*.tsv')):
        scores_df = pd.read_csv(file_s)
        scores_list.append(scores_df)

    items_all = pd.concat(items_list, axis=0).reset_index(drop=True)
    scores_all = pd.concat(scores_list, axis=1)
    return items_all, scores_all


def clean_items(items, scores):
    drop_list = []

    for index, row in items.iterrows():
        if pd.isna(row['scale']):
            drop_list.append(row['id'])
            items = items.drop(index)

    items_clean = items.reset_index(drop=True)
    items_clean['scale'] = items_clean['scale'].replace('_', ' ')
    scores = scores.apply(pd.to_numeric, errors='coerce')
    scores_clean = scores.drop(drop_list, axis=1).astype('float32')

    return items_clean, scores_clean


def construct_corr_df(items, pearson, spearman):
    schema = {'inventory_i': 'str', 'scale_i': 'str', 'item_i': 'str', 'id_i': 'str', 'inventory_j': 'str',
              'scale_j': 'str', 'item_j': 'str', 'id_j': 'str', 'pearson': 'float32', 'spearman': 'float32'}

    df = pd.DataFrame(columns=schema.keys()).astype(schema)

    df_size = math.comb(len(items), 2)

    df = pd.concat([df] * df_size, ignore_index=True)

    inventory_i = []
    scale_i = []
    item_i = []
    id_i = []
    inventory_j = []
    scale_j = []
    item_j = []
    id_j = []
    pearson_val = []
    spearman_val = []

    for idx_i in range(len(items) - 1):
        print(f'{idx_i + 1} of {len(items)}')
        for idx_j in range(idx_i + 1, len(items)):
            inventory_i.append(items['inventory'][idx_i])
            scale_i.append(items['scale'][idx_i])
            item_i.append(items['item'][idx_i])
            id_i.append(items['id'][idx_i])
            inventory_j.append(items['inventory'][idx_j])
            scale_j.append(items['scale'][idx_j])
            item_j.append(items['item'][idx_j])
            id_j.append(items['id'][idx_j])
            pearson_val.append(pearson[items['id'][idx_i]].loc[items['id'][idx_j]])
            spearman_val.append(spearman[items['id'][idx_i]].loc[items['id'][idx_j]])

    main_list = [inventory_i, scale_i, item_i, id_i, inventory_j, scale_j, item_j, id_j, pearson_val, spearman_val]

    for col_names, values_list in zip(schema.keys(), main_list):
        df[col_names] = values_list

    return df


if __name__ == "__main__":
    root = '/Users/josechonay/Library/CloudStorage/OneDrive-CarlvonOssietzkyUniversitätOldenburg/Winter Semester ' \
           '23-24/Internship/ARC'
    data_set = 'Open_Source_Psychometrics'
    scores_folder = os.path.join(root, data_set, 'scores')
    item_folder = os.path.join(root, data_set, 'items')

    items, scores = get_df(item_folder, scores_folder)
    print('data appended')

    items_clean, scores_clean = clean_items(items, scores)
    print('scores cleaned')

    pearson = scores_clean.corr(method='pearson', numeric_only=True).astype('float32')
    metadata_pearson = {'columns': pearson.columns.tolist(), 'index': pearson.index.tolist()}
    pearson.to_csv((os.path.join(root, data_set) + '/pearson.tsv'), header=None, index=False)
    with open((os.path.join(root, data_set) + '/pearson.json'), 'w') as metadata_p_file:
        json.dump(metadata_pearson, metadata_p_file)
    spearman = scores_clean.corr(method='spearman', numeric_only=True).astype('float32')
    metadata_spearman = {'columns': pearson.columns.tolist(), 'index': pearson.index.tolist()}
    spearman.to_csv((os.path.join(root, data_set) + '/spearman.tsv'), header=None, index=False)
    with open((os.path.join(root, data_set) + '/spearman.json'), 'w') as metadata_s_file:
        json.dump(metadata_spearman, metadata_s_file)
    print('correlations calculated')

    items_clean.to_csv((os.path.join(root, data_set) + '/items.tsv'), index=False)
    corr_df = construct_corr_df(items_clean, pearson, spearman)
    metadata_corr = {'columns': corr_df.columns.tolist()}
    corr_df.to_csv((os.path.join(root, data_set) + '/item_correlation.tsv'), index=False)
    with open((os.path.join(root, data_set) + '/item_correlation.json'), 'w') as metadata_c_file:
        json.dump(metadata_corr, metadata_c_file)
    # corr_df = pd.read_csv((os.path.join(root, data_set) + '/item_correlation.tsv'))
