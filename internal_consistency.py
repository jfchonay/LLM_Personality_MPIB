import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def clean_items(value):
    # clean our items by making sure all of them are strings. We remove any character that it's not a letter, number
    # or significant punctuation
    return re.sub('[^A-Za-z0-9.,!?()"\'\s:-]', ' ', str(value))


def get_similarities(correlations_df):
    # the first step is to divided our data frame into inventories, as we wanna only check the relationship of the
    # scales for the same inventory
    group_inventory = []
    # using the function groupby we can get a combination of all possible inventories in our data frame, and we can
    # create and identifier, so we know when the same inventory is being paired
    for inventory_name, inventory_df in correlations_df.groupby(['inventory_i', 'inventory_j'], sort=False):
        # our variable is going to store the name of the two inventories used in this group
        inventory_pairs = [inventory_name[0], inventory_name[1]]
        # we save that variable with its corresponding data frame
        group_inventory.append((inventory_pairs, inventory_df))
    # we are only interested in the data frames where the inventory_i and inventory_j are the same value,
    # so we filter our list, we must remember our data is a list of tuple, and we want to compare the first
    # value of the tuple
    filtered_list = [item for item in group_inventory if item[0][0] == item[0][1]]
    # now we can just retain the data frames we selected in the form of a list
    same_inventory = [only_df[1] for only_df in filtered_list]
    # we want to apply the same principle for scales, we want to grab all possible combinations of scales and create
    # a new set of the data frame for each possible combination
    final_combinations = []
    # we iterate over our list of data frames, we know each element of the list represents one data frame
    for inventory_df in same_inventory:
        group_scale = []
        # using the function groupby we can again create groups based on all possible combination of scales,
        # but we must pay attention as the order in which they appear makes them a unique combination
        scale_combinations = inventory_df.groupby(['scale_i', 'scale_j'], sort=False)
        for scale_name, scale_df in scale_combinations:
            # we take the identifier of the scales and sort them alphabetically, so we can easily compare
            # this variable later
            scale_sorted = sorted(scale_name)
            scale_pairs = scale_sorted[0] + scale_sorted[1]
            # we store each segmented data frame with their identifier as a tuple
            group_scale.append((scale_pairs, scale_df))
        # create an empty dictionary to store our identifiers and their respective data frames
        scale_dict = {}
        # we are going to iterate over the list of tuples we created
        for string, df in group_scale:
            # if there is an identifier that it's new we would add the key, value pair into our dictionary
            if string not in scale_dict:
                scale_dict[string] = df
            # now if one of our identifiers or keys is already in the dictionary we can use the same key and
            # concatenate the data frames
            else:
                scale_dict[string] = pd.concat([scale_dict[string], df], ignore_index=True)
        # we save all the dictionaries created into a list
        final_combinations.append(scale_dict)
    # to be efficient we can define a schema for our final data frame with column names and data types
    schema = {'inventory': 'str', 'scale_i': 'str', 'scale_j': 'str',
              'pearson_similarity': 'float32', 'spearman_similarity': 'float32'}
    final_df = pd.DataFrame(columns=schema.keys()).astype(schema)
    # create empty lists of the values we are going to extract from our dictionaries
    inventory = []
    scale_i = []
    scale_j = []
    pearson_simi = []
    spearman_simi = []
    # in our list every dictionary corresponds to one inventory, we iterate over all of them
    for inventory_dict in final_combinations:
        # the key is the string identifier and the value is the data frame of the corresponding scale pair
        for key, value in inventory_dict.items():
            # use data frame indexing to extract the values for each list, this should match our columns
            inventory.append(value['inventory_i'].values[0])
            scale_i.append(value['scale_i'].values[0])
            scale_j.append(value['scale_j'].values[0])
            pearson_simi.append(np.mean(value['pearson'].abs()))
            spearman_simi.append(np.mean(value['spearman'].abs()))
    # store everything into a main list
    main_list = [inventory, scale_i, scale_j, pearson_simi, spearman_simi]
    # construct our data frame using the schema and list to optimize the process
    for col_names, values_list in zip(schema.keys(), main_list):
        final_df[col_names] = values_list
    return final_df


def get_scale_zscore(similarities_df):
    # we want to calculate each inventory individually, as we want the mean and standard deviation of each inventory
    inventory_zscore = []
    # we can use the groupby function to segment our data
    for inventory_name, inventory_df in similarities_df.groupby('inventory', sort=False):
        # based on our data structure we know that we have two similarity measures, one based on pearson and one based
        # on spearman. We want to use them both but separate to calculate the z values
        select_cols = ['pearson_similarity', 'spearman_similarity']
        for col in select_cols:
            # calculate the z score and save it in a new column
            col_zscore = col + '_zscore'
            inventory_df[col_zscore] = (inventory_df[col] - inventory_df[col].mean()) / inventory_df[
                col].std(ddof=0)
        inventory_zscore.append(inventory_df)
    # we save one data frame for the full data set, that contains each individual inventory
    zscore_df = pd.concat(inventory_zscore, axis=0)
    return zscore_df


def plot_zscore(zscore_df, path, data_set):
    # for visualization, we only want to select the values of the z scores when the similarity is between the same
    # scale so we can use logical indexing for this
    scales_df = zscore_df[zscore_df['scale_i'] == zscore_df['scale_j']].reset_index(drop=True)
    # to facilitate plotting we are going to create a new ID column where we combine the scale name and inventory name
    # so that we have unique IDs for every pair
    scales_df['scale_inventory'] = scales_df['scale_i'] + scales_df['inventory']
    # again based on our data schema, we want to plot the pearson and spearman z values
    columns_plot = ['pearson_similarity_zscore', 'spearman_similarity_zscore']
    # we iterate over our desired values
    for col in columns_plot:
        plt.figure(figsize=(30, 20), dpi=1000)
        # we are going to use the unique ID as the x axis, the z score as the y axis, and we group them together by
        # the inventory
        ax = sns.barplot(scales_df, x='scale_inventory', y=col, hue='inventory', palette='Dark2', saturation=1,
                         dodge=True, legend='full', width=2)
        ax.set_title(f'Z values for the averaged similarity scores of items in the same scale \n '
                     f'for each inventory in {data_set} data set', fontsize=20)
        ax.set_ylabel('Z Score', fontsize=18)
        ax.set_xlabel('Scales', fontsize=18)
        ax.set_xticks([])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        # we want to add a horizontal line at value 1 and value 2, we expect all our values to be larger than 1 and
        # ideally larger than 2
        ax.axhline(y=1, xmin=0, xmax=1, linewidth=1, color='r')
        ax.axhline(y=2, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig((os.path.join(path, data_set, (col+'.png'))))
        plt.close()


if __name__ == "__main__":
    root = '/Users/josechonay/Library/CloudStorage/OneDrive-CarlvonOssietzkyUniversitätOldenburg/Winter Semester ' \
           '23-24/Internship/ARC'
    data_set = 'Open_Source_Psychometrics'
    data = 'item_correlation'

    correlations = pd.read_csv((os.path.join(root, data_set, data) + '.tsv'), header=0)
    correlations['item_i'] = correlations['item_i'].apply(clean_items)
    correlations['item_j'] = correlations['item_j'].apply(clean_items)

    paired_similarities_df = get_similarities(correlations)
    paired_similarities_df.to_csv((os.path.join(root, data_set) + '/similarity_scores.tsv'), index=False)

    zscore_df = get_scale_zscore(paired_similarities_df)

    plot_zscore(zscore_df, root, data_set)
