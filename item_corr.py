import pandas as pd
import os
import glob
import math
import json
import re


def get_df(item_path, scores_path, file_format):
    """This function will concatenate all the data frames inside the inventory directory that contain the item
    information and the scores' information. The item data frames should have as columns: inventory, scale, item and id.
    The scores' data frames should just have a unique id for every column. The function will check that there is a
    unique ID for every item and every column of the scores, if not it will raise an error, as this is a crucial
    requirement for the pipeline to work. It will return a data frame containing all items and another one for
    the scores.

    Parameters:
        item_path (str): the path to the folder containing the item data frames, one data frame per inventory
        scores_path (str): the path to the folder containing the scores data frames, one data frame per inventory
        file_format (str): the extension or file format for all data frames

    Returns:
        df
            items_all: a data frame containing all the items in the data set, with the structure: inventory, scale,
            item, id
        df
            scores_all: a data frame containing all the item scores for all inventories
   """
    # we are going to use the paths for the items data frame and the scores data frame to append the information into
    # one data frame so we can calculate the correlations between all items existing in the data set
    items_list = []
    # we are assuming all our information is saved in the same file format and that the header is in the first line
    for file_i in glob.iglob((item_path + '/*.' + file_format)):
        item_df = pd.read_csv(file_i, header=0)
        items_list.append(item_df)
    scores_list = []
    for file_s in glob.iglob((scores_path + '/*.' + file_format)):
        scores_df = pd.read_csv(file_s, header=0)
        scores_list.append(scores_df)
    # use the native pandas function to transform lists to data frames
    # in this case we assume all of our data frames have the same column names and number of columns
    items_all = pd.concat(items_list, axis=0).reset_index(drop=True)
    scores_all = pd.concat(scores_list, axis=1)
    # we want to check if all scores have a unique ID and that they match the ones in the items data frame
    if len(scores_all.columns.unique().values) != len(scores_all.columns.values):
        raise TypeError('The ID for each item in the scores data frames are not unique')
    elif sorted(scores_all.columns.values) != sorted(items_all['id'].values):
        raise TypeError('The ID for each item does not match the ID for the scores')
    return items_all, scores_all


def just_language(value):
    # clean our items by making sure all of them are strings. We remove any character that it's not a letter, number
    # or significant punctuation
    return re.sub('[^A-Za-z0-9.,!?()"\'\s:-]', ' ', str(value))


def clean_items(items, scores):
    """This function will clean the items and scores data frame, to make sure there are not any missing values that can
     interfere with the calculations, it will also remove any information of the items that it's not part of the
     language component of the item. It will return the cleaned data frames and a dictionary with the metadata so it
     can be saved as JSON file.

    Parameters:
        items (df): a data frame containing all the items in the data set, the columns should be: inventory, scale,
        item, id.
        scores (df): a data frame containing all the items and their scores, every row represents a subject and every
        column represents an item. Each item has a unique ID.

    Returns:
        df
            items_clean: a data frame containing all the items in the data set, with the structure: inventory, scale,
            item, id. It will only contain all items that are valid for calculating correlations, so they don't have
            any non values. It will contain only language elements for each item.
        dict
            items_md: a dictionary with the description of the data frame and the columns, made to be saved as json
            file with the data frame
        df
            scores_clean: a data frame containing all the item scores for all inventories that are valid for the
            calculation of correlation scores.
        dict
            scores_md: a dictionary with the description of the data frame and the columns, made to be saved as json
            file with the data frame
   """
    # our data set may contain elements that have nan values, which cannot be used to perform calculations we want
    # to drop this information
    drop_list = []
    # we are taking advantage that every item has an ID that matches the scores data sheet, so we can use that to index
    # in our data sheet and drop this elements
    for index, row in items.iterrows():
        if pd.isna(row['scale']):
            drop_list.append(row['id'])
            items = items.drop(index)
    # we can save our data and apply some last cleaning steps to remove any information that may not be part of the
    # items text
    items_clean = items.reset_index(drop=True)
    items_clean['scale'] = items_clean['scale'].replace('_', ' ')
    items_clean['item'] = items_clean['item'].apply(just_language)
    # transform all scores to number and 'float32' so we can preserve some memory
    scores = scores.apply(pd.to_numeric, errors='coerce')
    scores_clean = scores.drop(drop_list, axis=1).astype('float32')
    # create the meta data for each data frame
    items_md = {
        'information': 'This data frame contains all the items for all inventories in the data set, and a unique id '
                       'for each',
        'columns': items_clean.columns.tolist()
    }
    scores_md = {
        'information': 'This data frame contains all the scores for all items in the data set, and a unique id '
                       'for each that it is saved in the columns',
        'columns': scores_clean.columns.tolist()
    }
    return items_clean, items_md, scores_clean, scores_md


def construct_corr_df(items, scores, save, *path):
    """This function will construct a data frame that will contain all possible and unique pair wise correlation between
    all the items in the data set. In the function it will use the built-in pandas correlation calculation to calculate
    the matrix of all possible correlations between the scores. Then based on the structure of the items and scores
    data frame, where each item has a unique ID, it will index and construct all possible and unique pairings for
    every item. It is possible to toggle the save option to store the pearson and spearman correlation matrix in the
    local disk. The function will return a data frame with the information for inventory, scale and item for every pair
    of items and their calculated pearson and spearman values. It will look like: inventory_i, scale_i, item_i,
    inventory_j, scale_j, item_j, pearson, spearman. It will also return a dictionary that contains the metadata of the
    data frame and can be saved as a JSON file.

        Parameters:
            items (df): a data frame containing all the items in the data set, the columns should be: inventory, scale,
            item, id.
            scores (df): a data frame containing all the items and their scores, every row represents a subject and every
            column represents an item. Each item has a unique ID.
            save (bool): TRUE if you want to save the correlation matrix to your local disk
            path (str): if save is TRUE you should define the path were you want to save the data frame and its
            corresponding JSON file.

        Returns:
            df
                corr_df: a data frame containing all the pair wise possible and unique correlations between all the
                items in the data set. With the columns: inventory_i, scale_i, item_i, id_i, inventory_j, scale_j,
                item_j, id_j, pearson, spearman.
            dict
                corr_md: a dictionary with the description of the data frame and the columns, made to be saved as json
                file with the data frame
       """
    # use the built in pandas function to calculate correlation, so we preserve the ID as the column and index which
    # will help index inside our function. Set to numeric only so all nan values are omitted, we don't want them to be
    # treated like 0
    pearson = scores.corr(method='pearson', numeric_only=True).astype('float32')
    metadata_pearson = {
        'information': 'This data frame contains all possible inter item correlations for the data set',
        'columns': pearson.columns.tolist(),
        'index': pearson.index.tolist()
    }
    spearman = scores.corr(method='spearman', numeric_only=True).astype('float32')
    metadata_spearman = {
        'information': 'This data frame contains all possible inter item correlations for the data set',
        'columns': pearson.columns.tolist(),
        'index': pearson.index.tolist()
    }
    if save:
        pearson.to_csv((path + '/pearson.tsv'), header=None, index=False)
        with open((path + '/pearson.json'), 'w') as metadata_p_file:
            json.dump(metadata_pearson, metadata_p_file)
        spearman.to_csv((path + '/spearman.tsv'), header=None, index=False)
        with open((path + '/spearman.json'), 'w') as metadata_s_file:
            json.dump(metadata_spearman, metadata_s_file)
    # in order to preserve memory and be efficient in the creation of our new data frame we can set up a schema for the
    # new data frame with the data types we will be using
    schema = {'inventory_i': 'str', 'scale_i': 'str', 'item_i': 'str', 'id_i': 'str', 'inventory_j': 'str',
              'scale_j': 'str', 'item_j': 'str', 'id_j': 'str', 'pearson': 'float32', 'spearman': 'float32'}
    # apply the schema to the data frame and then allocate the total space we will need in it, we can calculate the
    # space because we know we want all possible non repeating combinations of items
    corr_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    # calculate all possible non repeating combinations of items
    corr_df_size = math.comb(len(items), 2)
    corr_df = pd.concat([corr_df] * corr_df_size, ignore_index=True)
    # determine the lists that will contain the information of our final data frame
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
    # we will iterate item by item and check all its possible combinations, so we start with the first position in the
    # items data frame and we don't need the last element as it will already be included as part of all possible
    # combinations with the other items
    for idx_i in range(len(items) - 1):
        # we want to iterate over all other items but the item itself
        for idx_j in range(idx_i + 1, len(items)):
            # this logical indexing works because our data frame spearman and pearson was created using the buitl in
            # function of pandas, so the column and index values are the same as the ID values of the items data frame
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
    # create a list that contains all of our information
    main_list = [inventory_i, scale_i, item_i, id_i, inventory_j, scale_j, item_j, id_j, pearson_val, spearman_val]
    # iterate over our schema and unpack the list we created to efficiently create our data frame
    for col_names, values_list in zip(schema.keys(), main_list):
        corr_df[col_names] = values_list
    # create a dictionary to store the meta data of our data frame
    corr_md = {
        'information': 'This data frame contains all possible and unique inter item correlations for all items in the'
                       ' data set',
        'columns': corr_df.columns.tolist()
    }
    return corr_df, corr_md


if __name__ == "__main__":
    root = 'path to folder'
    data_set = 'name of data set'
    scores_folder = os.path.join(root, data_set, 'scores')
    item_folder = os.path.join(root, data_set, 'items')

    items, scores = get_df(item_folder, scores_folder, 'tsv')

    items_clean, items_md, scores_clean, scores_md = clean_items(items, scores)
    items_clean.to_csv((os.path.join(root, data_set) + '/items_clean.tsv'), index=False)
    with open((os.path.join(root, data_set) + '/items_clean.json'), 'w') as metadata_c_file:
        json.dump(items_md, metadata_c_file)

    scores_clean.to_csv((os.path.join(root, data_set) + '/scores_clean.tsv'), index=False)
    with open((os.path.join(root, data_set) + '/scores_clean.json'), 'w') as metadata_c_file:
        json.dump(scores_md, metadata_c_file)

    corr_df, corr_md = construct_corr_df(items_clean, scores_clean, False)

    corr_df.to_csv((os.path.join(root, data_set) + '/item_correlation.tsv'), index=False)
    with open((os.path.join(root, data_set) + '/item_correlation.json'), 'w') as metadata_c_file:
        json.dump(corr_md, metadata_c_file)
