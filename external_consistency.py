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
                calculate the cosine similarity matrix, the second element is the cosine similarity matrix saved as
                 a tensor.
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
        embeddings = model.encode(sentences, convert_to_tensor=True)
        # calculate the cosine similarity between all embeddings, this matrix will be of the size of
        # embeddings X embeddings and will contain all similarities
        all_cosine = util.cos_sim(embeddings, embeddings)
        # now we can create every row of our data frame as an element of the list
        item_cosine.append((item_id, all_cosine))
    return item_cosine


def lt_corr_similarities(cosine_similarity, correlations):
    # we will define the elements of our final data frame in lists so the creation can be efficient
    scale_id = []
    inventory_id = []
    cosine_pearson = []
    cosine_spearman = []
    # because our cosine similarity variable is a list of tuples we will use this structure to construct the data
    for one_scale in cosine_similarity:
        # in case there are scales with only one element we want to skip them
        if len(one_scale[0]) == 1:
            continue
        else:
            # access the information stored in the tuple, one is the list of IDs we used to create the embeddings, and
            # in the same order of that list we want to construct a matrix that represents the correlations calculated
            item_id = one_scale[0]
            scale_cosine = one_scale[1]
            matrix_pearson = []
            matrix_spearman = []
            # iterate over all elements of the item ID's to access the correct correlation calculation
            for i_p in range(len(item_id)):
                # every row should represent the paired correlation between item i_p and all other items, and 1 in the
                # diagonals as that is the value of correlation between themselves
                row_p = [correlations.loc[(correlations['id_i'] == item_id[i_p]) &
                                          (correlations['id_j'] == item_id[j_p])]['pearson'].values if j_p != i_p
                         else 1 for j_p in range(len(item_id))]
                matrix_pearson.append(row_p)
            # repeat the same process for the spearman correlation
            for i_s in range(len(item_id)):
                row_s = [correlations.loc[(correlations['id_i'] == item_id[i_s]) &
                                          (correlations['id_j'] == item_id[j_s])]['spearman'].values if j_s != i_s
                         else 1 for j_s in range(len(item_id))]
                matrix_spearman.append(row_s)
            # now we want to extract only the lower triangle of our matrix, excluding the diagonal
            lower_triangle_cosine = [scale_cosine[i_c][j_c] for i_c in range(len(scale_cosine)) for j_c in
                                     range(i_c + 1, len(scale_cosine[i_c]))]
            # we want only the absolute value of our data
            lower_triangle_cosine = [abs(cs) for cs in lower_triangle_cosine]
            # extract the triangle and only absolute value for our pearson matrix
            lower_triangle_pearson = [matrix_pearson[i_mp][j_mp] for i_mp in range(len(matrix_pearson)) for j_mp in
                                      range(i_mp + 1, len(matrix_pearson[i_mp]))]
            lower_triangle_pearson = [abs(mp) for mp in lower_triangle_pearson]
            # extract the triangle and only absolute value for our spearman matrix
            lower_triangle_spearman = [matrix_spearman[i_ms][j_ms] for i_ms in range(len(matrix_spearman)) for j_ms in
                                       range(i_ms + 1, len(matrix_spearman[i_ms]))]
            lower_triangle_spearman = [abs(ms) for ms in lower_triangle_spearman]
            # now we can calculate the sepearman correlation between our cosine and pearson and spearman measures
            corr_cos_pear = spearmanr(lower_triangle_cosine, lower_triangle_pearson, nan_policy='omit').statistic
            corr_cos_spr = spearmanr(lower_triangle_cosine, lower_triangle_spearman, nan_policy='omit').statistic
            # append each value to use for our data frame, we will get one pearson and spearman per scale per inventory
            inventory_id.append(correlations.loc[correlations['id_i'] == item_id[0]]['inventory_i'].unique()[0])
            scale_id.append(correlations.loc[correlations['id_i'] == item_id[0]]['scale_i'].unique()[0])
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


def plot_corr_lt(correlations_df, path, data_set):
    # clean our data frame by deleting any empty values, normally created by uni dimensional tests
    correlations_df = correlations_df.dropna()
    # we know we want to plot for each similarity relationship, cosine-pearson and cosine-spearman
    corr_types = ['pearson', 'spearman']
    for simi_type in corr_types:
        plt.figure(figsize=(30, 20), dpi=1000)
        ax = sns.scatterplot(correlations_df, x=correlations_df['scale'], y=correlations_df[f'cosine_{simi_type}'],
                             hue=correlations_df['inventory'], hue_order=correlations_df['inventory'],
                             palette='tab10', s=10)
        ax.axhline(y=0.2, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
        ax.set_ylim(-1, 1.1)
        ax.set_title(f'Between scale Spearman correlation between the cosine and {simi_type} similarity, \n '
                     f'for all inventories in the {data_set} dataset', fontsize=20)
        ax.set_ylabel('Spearman coefficient', fontsize=16)
        ax.set_xlabel('Scales', fontsize=16)
        ax.set_xticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig((os.path.join(path, data_set, ('lt_cosine_' + simi_type + '.png'))))
        plt.close()


def loo_corr_similarities(cosine_similarity, empirical_similarity):
    # we want to keep only the values that represent the similarity between the same scale, as in this data frame we
    # have calculated the similarity between all possible scales in the same inventory
    empirical_similarity = empirical_similarity[empirical_similarity['scale_i'] ==
                                                empirical_similarity['scale_j']].reset_index(drop=True)
    # we can merge our two data frames, as they will have the same value in inventory and the same value in scale
    all_similarities = pd.merge(cosine_similarity, empirical_similarity, left_on=['inventory', 'scale'],
                                right_on=['inventory', 'scale_i']).reset_index(drop=True)
    df = all_similarities.drop(['scale_j', 'scale_i'], axis=1)
    all_inventory = []
    final_pearson = []
    final_spearman = []
    inventory_ID = []
    # iterate over every individual inventory to create a list of data frames by scale, we want to reset the index
    # for simplifying indexing in the next steps
    for inventory_name, inventory_df in df.groupby('inventory', sort=False):
        all_inventory.append(inventory_df.reset_index(drop=True))
    # now we can iterate over each inventory data frame and calculate the relationship between the scales
    # different similarity measurements
    for one_inventory in all_inventory:
        inventory_correlation_pearson = []
        inventory_correlation_spearman = []
        for i in range(len(one_inventory)):
            # Omit the ith value in both predicted and empirical similarities
            loo_inventory = one_inventory.drop(i)
            correlation_pearson = spearmanr(loo_inventory['cosine_similarity'].abs(),
                                            loo_inventory['pearson_similarity'].abs(), nan_policy='omit').statistic
            correlation_spearman = spearmanr(loo_inventory['cosine_similarity'].abs(),
                                             loo_inventory['spearman_similarity'].abs(), nan_policy='omit').statistic
            # we calculate the spearman correlation for every row or scale in an inventory while omiting that scale
            inventory_correlation_pearson.append(correlation_pearson)
            inventory_correlation_spearman.append(correlation_spearman)
        # construct the lists with the elements for each inventory
        inventory_ID.append(one_inventory['inventory'].unique()[0])
        final_pearson.append(np.mean(inventory_correlation_pearson))
        final_spearman.append(np.mean(inventory_correlation_spearman))
    # define the column names and data types for our final data frame
    schema = {'inventory': 'str', 'scale': 'str', 'cosine_pearson': 'float32', 'cosine_spearman': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    main_list = [inventory_ID, final_pearson, final_spearman]
    # construct the final data frame by unzipping the information in our lists and dictionary
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list
    return final_df


def plot_corr_loo(correlations_df, path, data_set):
    # clean our data frame by deleting any empty values, normally created by uni dimensional tests
    correlations_df = correlations_df.dropna()
    # we know we want to plot for each similarity relationship, cosine-pearson and cosine-spearman
    corr_types = ['pearson', 'spearman']
    for simi_type in corr_types:
        plt.figure(figsize=(20, 10), dpi=1000)
        ax = sns.scatterplot(correlations_df, x=correlations_df['inventory'], y=correlations_df[f'cosine_{simi_type}'],
                             hue=correlations_df['inventory'], hue_order=correlations_df['inventory'],
                             palette='tab10', s=200)
        ax.set_title(f'Leave one out Spearman correlation between the cosine and {simi_type} similarity, \n '
                     f'for all inventories in the {data_set} dataset', fontsize=20)
        ax.set_ylim(-1, 1.1)
        ax.set_ylabel('Spearman coefficient', fontsize=16)
        ax.set_xlabel('Inventory', fontsize=16)
        ax.set_xticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig((os.path.join(path, data_set, ('loo_cosine_' + simi_type + '.png'))))
        plt.close()


if __name__ == "__main__":
    root = '/Users/josechonay/Library/CloudStorage/OneDrive-CarlvonOssietzkyUniversitätOldenburg/Winter Semester ' \
           '23-24/Internship/ARC'
    data_set = 'Springfield_Community'

    items = pd.read_csv(os.path.join(root, data_set, 'items.tsv'), header=0)

    cosine_similarity = scale_cos_sim(items)

    correlations = pd.read_csv((os.path.join(root, data_set) + '/item_correlation.tsv'), header=0)

    lt_similarities_corr, lt_md = lt_corr_similarities(cosine_similarity, correlations)
    lt_similarities_corr.to_csv((os.path.join(root, data_set) + '/spearman_predicted_empirical_similarity.tsv'),
                                index=False)
    with open((os.path.join(root, data_set) + '/spearman_predicted_empirical_similarity.json'), 'w') as metadata_file:
        json.dump(lt_md, metadata_file)

    plot_corr_lt(lt_similarities_corr, root, data_set)
    #
    # loo_similarities_corr = loo_corr_similarities(cosine_similarity, empirical_similarity)
    #
    # plot_corr_loo(loo_similarities_corr, root, data_set)
    # test_data = all_similarities[all_similarities['inventory'] == '275-IPIP']
    # correlation_test = spearmanr(test_data['cosine_similarity'], test_data['pearson_similarity'],
    #                              nan_policy='omit').statistic
    # # Number of permutations
    # num_permutations = 1000
    #
    # # Perform permutation test
    # permuted_correlations = []
    # for _ in range(num_permutations):
    #     # Permute the second array
    #     permuted_array2 = np.random.permutation(test_data['pearson_similarity'])
    #
    #     # Calculate Spearman correlation for the permuted data
    #     permuted_corr, _ = spearmanr(test_data['cosine_similarity'], permuted_array2, nan_policy='omit')
    #     permuted_correlations.append(permuted_corr)
    #
    # # Plot the distribution of permuted correlations
    # plt.hist(permuted_correlations, bins=30, color='blue', alpha=0.7, label='Permuted Correlations')
    # plt.axvline(correlation_test, color='red', linestyle='dashed', linewidth=2, label='Observed Correlation')
    # plt.title('Permutation Test for Spearman Correlation')
    # plt.xlabel('Spearman Correlation')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.show()
