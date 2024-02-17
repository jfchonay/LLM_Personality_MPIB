from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import json


def scale_cos_sim(items_df):
    """This function will use the sentence transformer MPnet to calculate the cosine similarity of items in the same
    scale for all inventories in the data set. It takes a data frame containing only the language portions of the items,
    and that it has the following columns: inventory, scale, item, id. It will return a list of tuples, the first
    element of the tuple is the list of item ID, and the second is the matrix containing the similarities.

        Parameters:
            items_df (df): a data frame that contains all items belonging to the data set, has the columns: inventory,
            scale, item, id.

        Returns:
            list
                items_cosine: a list of tuples, the first element of the tuple is the id of the sentences used to
                calculate the cosine similarity matrix, the second element is the inventory name as a string, the third
                element is the scale name as a string and the fourth element is the cosine similarity matrix saved as a
                tensor.
       """
    # we define our model for the sentence transformer, we are using MPnet. For more information look up for the
    # documentation in sbert.net
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    group_inventory = []
    # we are constructing similarity measures between the scales of the same inventories, so we want to segment our
    # data by inventories
    for inventory_name, inventory_df in items_df.groupby('inventory', sort=False):
        group_inventory.append(inventory_df)
    # define column names and data types for our data frame, also the lists that will store the data
    group_scale = []
    item_cosine = []
    # iterate over every individual inventory to create a list of data frames by scale, we want to reset the index
    # for simplifying indexing in the next steps
    for inventory_df in group_inventory:
        scale_combinations = inventory_df.groupby('scale', sort=False)
        for scale_name, scale_df in scale_combinations:
            # we end with one data frame per scale in each inventory
            group_scale.append(scale_df.reset_index(drop=True))
    # now if we iterate over each element we are accessing individual scales for each inventory
    for individual_scale in group_scale:
        # we can extract the sentences for the model to create the embeddings, we want to extrac the item ID
        sentences = individual_scale['item'].tolist()
        item_id = individual_scale['id'].tolist()
        inventory = individual_scale['inventory'].unique()[0]
        scale = individual_scale['scale'].unique()[0]
        embeddings = model.encode(sentences, convert_to_tensor=True)
        # calculate the cosine similarity between all embeddings, this matrix will be of the size of
        # embeddings X embeddings and will contain all similarities and take the absolute value of it
        all_cosine = abs(util.cos_sim(embeddings, embeddings))
        # now we can create every row of our data frame as an element of the list
        item_cosine.append((item_id, inventory, scale, all_cosine))
    return item_cosine


def lt_corr_similarities(cosine_similarity, correlations):
    """This function will calculate the scale spearman correlation between the cosine similarities and pearson and
    spearman similarities. The cosine similarities are calculated with the embeddings, and they are in the form of a
    list. The list contain tuples, in the first element there is the list of item IDs of that scale,the second
    element contains the name of the inventory, the third the name of the scale, and the fourth is the cosine similarity
    matrix between all those items. The correlation data frame contains all possible inter item correlation pairs for
    this data set. We want to extract the lower triangle of both similarities, meaning all the possible and unique
    pairings. Then the function will calculate the spearman correlation between both arrays and return a value per
    scale for all scales in all inventories in the data set. The resulting data frame will have columns: inventory,
    scale, cosine_pearson, spearman_pearson

        Parameters:
            cosine_similarity (list): a list of tuples, the first element of the tuple is the id of the sentences used
            to calculate the cosine similarity matrix, the second element is the inventory name as a string, the third
            element is the scale name as a string and the fourth element is the cosine similarity matrix saved as a
            tensor.
            correlations (df): a data frame containing all possible pair wise correlations for all items in the data
            set. It should have the columns: inventory_i, scale_i, item_i, id_i, inventory_j, scale_j, item_j, id_j,
            pearson, spearman.

        Returns:
            df
                final_df: a data frame that contains the spearman coefficient for every scale, between their cosine
                similarity and their pearson and spearman similarity. Has the columns: inventory, scale, cosine_pearson,
                cosine_spearman
            dict
                df_md: a dictionary containing the metadata of the data frame, a description and column names
       """
    # we will define the elements of our final data frame in lists so the creation can be efficient
    scale_id = []
    inventory_id = []
    cosine_pearson = []
    cosine_spearman = []
    # because our cosine similarity variable is a list of tuples we will use this structure to construct the data, every
    # element of the list represents a unique scale of an inventory
    for one_scale in cosine_similarity:
        # in case there are scales with only one element we want to skip them
        if len(one_scale[0]) < 3:
            continue
        else:
            # access the information stored in the tuple, one is the list of IDs we used to create the embeddings, and
            # in the same order of that list we want to construct a matrix that represents the correlations calculated
            item_id = one_scale[0]
            inventory = one_scale[1]
            scale = one_scale[2]
            scale_cosine = one_scale[3]
            # extract the section of the data frame that contains all inter item correlations for all items in one scale
            # for all inventories, extract only the pearson values and their absolute value
            pearson_scores = abs(correlations[(correlations['inventory_i'] == inventory) &
                                              (correlations['scale_i'] == scale) &
                                              (correlations['inventory_j'] == inventory) &
                                              (correlations['scale_j'] == scale)]['pearson'].values)
            # repeat the same process for the spearman correlation
            spearman_scores = abs(correlations[(correlations['inventory_i'] == inventory) &
                                          (correlations['scale_i'] == scale) &
                                          (correlations['inventory_j'] == inventory) &
                                          (correlations['scale_j'] == scale)]['spearman'].values)
            # now we want to extract only the lower triangle of our matrix, excluding the diagonal
            lower_triangle_cosine = [scale_cosine[i_c][j_c] for i_c in range(len(scale_cosine)) for j_c in
                                     range(i_c + 1, len(scale_cosine[i_c]))]
            # now we can calculate the sepearman correlation between our cosine and pearson and spearman measures
            corr_cos_pear = spearmanr(lower_triangle_cosine, pearson_scores, nan_policy='omit').statistic
            corr_cos_spr = spearmanr(lower_triangle_cosine, spearman_scores, nan_policy='omit').statistic
            # append each value to use for our data frame, we will get one pearson and spearman per scale per inventory
            inventory_id.append(inventory)
            scale_id.append(scale)
            cosine_pearson.append(corr_cos_pear)
            cosine_spearman.append(corr_cos_spr)
    # define the schema for our data frame, with the column  names and data types
    schema = {'inventory': 'str', 'scale': 'str', 'cosine_pearson': 'float32', 'cosine_spearman': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    main_list = [inventory_id, scale_id, cosine_pearson, cosine_spearman]
    # construct the final data frame by unzipping the information in our lists and dictionary
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list
    # define a dictionary with the meta data of our data frame
    df_md = {
        'information': 'This data frame contains the spearman correlation between the lower triangle of the cosine'
                       ' similarity matrix and the lower triangle of the spearman and pearson similarity matrix. '
                       'Correlation is calculated scale by scale for all inventories in the data set.',
        'columns': final_df.columns.tolist()
    }
    return final_df, df_md


def plot_corr_lt(correlations_df, path):
    """This function uses matplotlib and seaborn libraries to plot the spearman coefficients of the correlation
     between pearson and spearman and cosine similarity. It will save the plots to the path and data set that you set.

            Parameters:
                correlations_df (df): a data frame that contains scales spearman correlation in each inventory for all
                inventories in the data set and their z value. It should have the columns inventory, scale,
                cosine_pearson, cosine_spearman
                path (str): the path to the folder where you want to save the figures

           """
    # clean our data frame by deleting any empty values, normally created by uni dimensional tests
    correlations_df = correlations_df.dropna()
    # we want to add a unique identifier for every row, so we can group and use the seaborn barplot function
    correlations_df['scale_inventory'] = correlations_df['scale'] + correlations_df['inventory']
    # we know we want to plot for each similarity relationship, cosine-pearson and cosine-spearman
    corr_types = ['pearson', 'spearman']
    for simi_type in corr_types:
        plt.figure(figsize=(30, 20), dpi=1000)
        # we are going to use the unique ID as the x axis, the z score as the y axis, and we group them together by
        # the inventory
        ax = sns.barplot(correlations_df, x=correlations_df['scale_inventory'], y=correlations_df[f'cosine_{simi_type}'],
                         hue=correlations_df['inventory'], palette='Dark2', saturation=1, dodge=True, legend='full',
                         width=2)
        ax.set_ylim(-1, 1.1)
        ax.set_title(f'Between scale Spearman correlation between the cosine and {simi_type} similarity, \n '
                     f'for all inventories in the {data_set} dataset', fontsize=20)
        ax.set_ylabel('Spearman coefficient', fontsize=16)
        ax.set_xlabel('Scales', fontsize=16)
        ax.set_xticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig((os.path.join(path, ('lt_cosine_' + simi_type + '.png'))))
        plt.close()


def create_square_matrix(list_rows):
    m = len(list_rows[0])
    full_matrix = np.zeros((m, m))
    # Populate the matrix with values from the list
    for i, array in enumerate(list_rows):
        # Set the diagonal element
        full_matrix[i, i] = array[0]
        # Set values in the upper triangle
        full_matrix[i, i + 1:] = array[1:]
        # Mirror the values in the lower triangle
        full_matrix[i + 1:, i] = array[1:]
    np.fill_diagonal(full_matrix, 1)
    return full_matrix


def item_corr_similarities(cosine_similarity, correlations):
    """This function will calculate the scale spearman correlation between the cosine similarities and pearson and
        spearman similarities for every item, in every scale, for all inventories in the data set. It will extract the
        similarity scores of one item with all other items in the scale, excluding the relationship with itself. It will
        do this procedure for cosine similarity, pearson and spearman similarity. Then it will take the array of cosine
        similarity and calculate a spearman correlation with pearson and spearman. It will return a data frame that
        contains the columns: inventory, scale, item, cosine_pearson, cosine_spearman

            Parameters:
                cosine_similarity (list): a list of tuples, the first element of the tuple is the id of the sentences
                used to calculate the cosine similarity matrix, the second element is the inventory name as a string,
                the third element is the scale name as a string and the fourth element is the cosine similarity matrix
                 saved as a tensor
                correlations (df): a data frame containing all possible pair wise correlations for all items in the data
                set. It should have the columns: inventory_i, scale_i, item_i, id_i, inventory_j, scale_j, item_j, id_j,
                pearson, spearman.

            Returns:
                df
                    final_df: a data frame that contains the spearman coefficient for every item, between their cosine
                    similarity and their pearson and spearman similarity. Has the columns: inventory, scale, item,
                    cosine_pearson, cosine_spearman
                dict
                    df_md: a dictionary containing the metadata of the data frame, a description and column names
           """
    # we will define the elements of our final data frame in lists so the creation can be efficient
    scale_id = []
    inventory_id = []
    item = []
    cosine_pearson = []
    cosine_spearman = []
    # because our cosine similarity variable is a list of tuples we will use this structure to construct the data, every
    # element of the list represents a unique scale of an inventory
    for one_scale in cosine_similarity:
        # in case there are scales with only one or two element we want to skip them
        if len(one_scale[0]) < 3:
            continue
        else:
            # access the information stored in the tuple, one is the list of IDs we used to create the embeddings, and
            # in the same order of that list we want to construct a matrix that represents the correlations calculated
            item_id = one_scale[0]
            inventory = one_scale[1]
            scale = one_scale[2]
            scale_cosine = one_scale[3]
            slice_correlation = correlations[(correlations['inventory_i'] == inventory) &
                                              (correlations['scale_i'] == scale) &
                                              (correlations['inventory_j'] == inventory) &
                                              (correlations['scale_j'] == scale)]
            matrix_pearson = []
            matrix_spearman = []
            for i_p in range(len(item_id)-1):
                # we can extract all possible correlation values for our item, because of our data frame we know that
                # this will also represent each row in the cosine similarity matrix as the items are selected in the
                # same order
                row_p = abs(slice_correlation[slice_correlation['id_i'] == item_id[i_p]]['pearson'].values)
                # we add a one in the front as we know this is the correlation between the item and itself
                row_p = np.insert(row_p, 0, 1)
                # stack to end up with a 1D matrix
                matrix_pearson.append(np.hstack(row_p))
                # repeat for spearman
                row_s = abs(slice_correlation[slice_correlation['id_i'] == item_id[i_p]]['spearman'].values)
                row_s = np.insert(row_s, 0, 1)
                matrix_spearman.append(np.hstack(row_s))
            # we will end with a list containing each row of our square matrix, we want to now construct the square
            # matrix of correlations so we can iterate row by row for every item
            full_matrix_pearson = create_square_matrix(matrix_pearson)
            full_matrix_spearman = create_square_matrix(matrix_spearman)
            # we are going to iterate row by row, and we want to index into all items excluding the item that represents
            # that row
            for i_id in range(len(item_id)):
                # we will sum here al indexes before the item and all indexes after, having all possible indexes but
                # the item itself
                row_idx = list(range(len(item_id)))[:i_id] + list(range(len(item_id))[i_id+1:])
                # now we can use those indexes to extract each row of cosine similarity matrix, pearson and spearman.
                # This works because all 3 matrices are square and they are in the same order of items.
                one_row_pearson = [full_matrix_pearson[i_id][j_c] for j_c in row_idx]
                one_row_spearman = [full_matrix_spearman[i_id][j_c] for j_c in row_idx]
                one_row_cosine = [scale_cosine[i_id][j_c] for j_c in row_idx]
                # now we can calculate the spearman correlation between our cosine and pearson and spearman measures
                corr_cos_pear = spearmanr(one_row_cosine, one_row_pearson, nan_policy='omit').statistic
                corr_cos_spr = spearmanr(one_row_cosine, one_row_spearman, nan_policy='omit').statistic
                # append each value to use for our data frame, we will get one pearson and spearman per scale per
                # item, per scale, for all inventories
                inventory_id.append(inventory)
                scale_id.append(scale)
                # because of the structure of our correlations data frame, all pair correlation exist but only once,
                # so we want to check when we are dealing with an item that only exists in the column j of the data
                if slice_correlation.loc[slice_correlation['id_i'] == item_id[i_id]]['item_i'].unique().size == 0:
                    item.append(slice_correlation.loc[slice_correlation['id_j'] == item_id[i_id]]['item_j'].unique()[0])
                else:
                    item.append(slice_correlation.loc[slice_correlation['id_i'] == item_id[i_id]]['item_i'].unique()[0])
                cosine_pearson.append(corr_cos_pear)
                cosine_spearman.append(corr_cos_spr)
    # define the column names and data types for our final data frame
    schema = {'inventory': 'str', 'scale': 'str', 'item': 'str', 'cosine_pearson': 'float32',
              'cosine_spearman': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    main_list = [inventory_id, scale_id, item, cosine_pearson, cosine_spearman]
    # construct the final data frame by unzipping the information in our lists and dictionary
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list
    df_md = {
        'information': 'This data frame contains the spearman correlation between the similarity of each item with all'
                       'other items in its scale, excluding itself. The similarity measures are cosine, pearson and'
                       'spearman.',
        'columns': final_df.columns.tolist()
    }
    return final_df, df_md


if __name__ == "__main__":
    root = 'path to folder'
    data_set = 'Springfield_Community'

    items = pd.read_csv(os.path.join(root, data_set, 'items_clean.tsv'), header=0)

    cosine_similarity = scale_cos_sim(items)

    correlations = pd.read_csv((os.path.join(root, data_set) + '/item_correlation.tsv'), header=0)

    lt_similarities_corr, lt_md = lt_corr_similarities(cosine_similarity, correlations)
    lt_similarities_corr.to_csv((os.path.join(root, data_set) + '/spearman_lower_triangle_similarity.tsv'),
                                index=False)
    with open((os.path.join(root, data_set) + '/spearman_lower_triangle_similarity.json'), 'w') as metadata_file:
        json.dump(lt_md, metadata_file)

    plot_corr_lt(lt_similarities_corr, os.path.join(root, data_set))

    item_similarities, it_sim_md = item_corr_similarities(cosine_similarity, correlations)
    item_similarities.to_csv((os.path.join(root, data_set) + '/item_similarities_spearman.tsv'),
                             index=False)
    with open((os.path.join(root, data_set) + '/item_similarities_spearman.json'), 'w') as metadata_file:
        json.dump(it_sim_md, metadata_file)
