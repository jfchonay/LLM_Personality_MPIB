from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt


def scale_cos_sim(items_df):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    group_inventory = []
    for inventory_name, inventory_df in items_df.groupby('inventory', sort=False):
        group_inventory.append(inventory_df)
    schema = ['inventory', 'scale', 'cosine_similarity']
    final_df = pd.DataFrame(columns=schema)
    inventory = []
    scale = []
    cosine_similarity = []
    group_scale = []
    for inventory_df in group_inventory:
        scale_combinations = inventory_df.groupby('scale', sort=False)
        for scale_name, scale_df in scale_combinations:
            group_scale.append(scale_df.reset_index(drop=True))
    for individual_scale in group_scale:
        sentences = individual_scale['item'].tolist()
        embeddings = model.encode(sentences, convert_to_tensor=True)
        all_cosine = util.cos_sim(embeddings, embeddings)
        unique_cosine = []
        for i in range(len(all_cosine) - 1):
            for j in range(i + 1, len(all_cosine)):
                unique_cosine.append(all_cosine[i][j])
        inventory.append(individual_scale['inventory'][0])
        scale.append(individual_scale['scale'][0])
        cosine_similarity.append(np.mean(unique_cosine))
    main_list = [inventory, scale, cosine_similarity]
    for col_names, values_list in zip(schema, main_list):
        final_df[col_names] = values_list
    return final_df


def get_corr_similarities(cosine_similarity, empirical_similarity):
    empirical_similarity = empirical_similarity[empirical_similarity['scale_i'] ==
                                                empirical_similarity['scale_j']].reset_index(drop=True)

    all_similarities = pd.merge(cosine_similarity, empirical_similarity, left_on=['inventory', 'scale'],
                                right_on=['inventory', 'scale_i']).reset_index(drop=True)
    all_similarities = all_similarities.drop(['scale_j', 'scale_i'], axis=1)

    all_inventory = []
    for inventory_name, inventory_df in all_similarities.groupby('inventory', sort=False):
        all_inventory.append(inventory_df)
    inventory_ID = []
    inventory_correlation_pearson = []
    inventory_correlation_spearman = []
    for one_inventory in all_inventory:
        correlation_pearson = spearmanr(one_inventory['cosine_similarity'].abs(), one_inventory['pearson_similarity'].abs(),
                                        nan_policy='omit').statistic
        correlation_spearman = spearmanr(one_inventory['cosine_similarity'].abs(), one_inventory['spearman_similarity'].abs(),
                                         nan_policy='omit').statistic
        inventory_ID.append(one_inventory['inventory'].unique()[0])
        inventory_correlation_pearson.append(correlation_pearson)
        inventory_correlation_spearman.append(correlation_spearman)
    schema = {'inventory': 'str', 'cosine_pearson': 'float32', 'cosine_spearman': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    main_list = [inventory_ID, inventory_correlation_pearson, inventory_correlation_spearman]
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list

    return final_df


def plot_corr_lt(correlations_df, root, data_set):
    correlations_df = correlations_df.dropna()
    corr_types = ['pearson', 'spearman']
    for simi_type in corr_types:
        plt.figure(figsize=(20, 10), dpi=1000)
        ax = sns.scatterplot(correlations_df, x=correlations_df['inventory'], y=correlations_df[f'cosine_{simi_type}'],
                             hue=correlations_df['inventory'], hue_order=correlations_df['inventory'],
                             palette='tab10', s=200)
        ax.axhline(y=0.2, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
        ax.set_title(f'Lower triangle  Spearman correlation between the cosine and {simi_type} similarity, \n '
                     f'for all inventories in the {data_set} dataset', fontsize=20)
        ax.set_ylabel('Spearman coefficient', fontsize=16)
        ax.set_xlabel('Inventory', fontsize=16)
        ax.set_xticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig((os.path.join(root, data_set, ('sr_cosine_'+simi_type+'.png'))))
        plt.close()


if __name__ == "__main__":
    root = '/Users/josechonay/Library/CloudStorage/OneDrive-CarlvonOssietzkyUniversitätOldenburg/Winter Semester ' \
           '23-24/Internship/ARC'
    data_set = 'Springfield_Community'

    items = pd.read_csv(os.path.join(root, data_set, 'items.tsv'), header=0)

    cosine_similarity = scale_cos_sim(items)

    empirical_similarity = pd.read_csv(os.path.join(root, data_set, 'similarity_scores.tsv'), header=0)

    # similarities_corr = get_corr_similarities(cosine_similarity, empirical_similarity)
    #
    # plot_corr_lt(similarities_corr, root, data_set)

    empirical_similarity = empirical_similarity[empirical_similarity['scale_i'] ==
                                                empirical_similarity['scale_j']].reset_index(drop=True)
    all_similarities = pd.merge(cosine_similarity, empirical_similarity, left_on=['inventory', 'scale'],
                                right_on=['inventory', 'scale_i']).reset_index(drop=True)
    df = all_similarities.drop(['scale_j', 'scale_i'], axis=1)

    all_inventory = []
    final_pearson = []
    final_spearman = []
    inventory_ID = []
    for inventory_name, inventory_df in df.groupby('inventory', sort=False):
        all_inventory.append(inventory_df)
    for one_inventory in all_inventory:
        inventory_correlation_pearson = []
        inventory_correlation_spearman = []
        for i in range(len(one_inventory)):
            # Omit the ith value in both predicted and empirical similarities
            loo_inventory = one_inventory.drop(i, axis=1)
            correlation_pearson = spearmanr(loo_inventory['cosine_similarity'].abs(),
                                            loo_inventory['pearson_similarity'].abs(), nan_policy='omit').statistic
            correlation_spearman = spearmanr(loo_inventory['cosine_similarity'].abs(),
                                             loo_inventory['spearman_similarity'].abs(), nan_policy='omit').statistic
            inventory_correlation_pearson.append(correlation_pearson)
            inventory_correlation_spearman.append(correlation_spearman)
        inventory_ID.append(one_inventory['inventory'].unique()[0])
        final_pearson.append(np.mean(inventory_correlation_pearson))
        final_spearman.append(np.mean(inventory_correlation_spearman))

    schema = {'inventory': 'str', 'cosine_pearson': 'float32', 'cosine_spearman': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    main_list = [inventory_ID, final_pearson, final_spearman]
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list

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
