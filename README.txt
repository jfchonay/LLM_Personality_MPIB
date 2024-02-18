This work was done to analise the contents of data sets of personality inventories, which were cleaned, organised and
manipulated for the purpose of calculating sentence embeddings. From the raw files downloaded from:
1) Springfield Community data base: https://dataverse.harvard.edu/dataverse/ESCS-Data
2) Open Source Psychometrics: https://openpsychometrics.org/_rawdata/
3) Kajonus & Johnson 2019: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7871748/
This information was used to create different files or data frames in format ‘.tsv’, this files contain all the
scores available for each inventory, only US scores were used. As the header of this files there is an item ID that
is unique between all inventories of the same data set, this was done for the analysis pipeline. Then multiple item data
frames in format ’.tsv’, each file contains a list of every item in the inventory, and to which scale they belong,
and a unique ID which corresponds to the one found in the scores files.
The analysis pipeline consisted of creating one file with all scores and items for the data set, and using the scores
 to calculate all possible inter item correlations. Then there was a process of data validation, internal and external.
 In the internal validation it was evaluated if the empirical similarity measure of items was higher within scale than
 between scale, this was done with the z values of the similarity measures. The external validation consisted on using
 sentence embeddings to calculate predicted similarity measures of items based on the cosine similarity of the
 embeddings. After that the Spearman correlation between predicted and empirical similarity was computed in two ways,
 one taking all possible and unique pairings in a scale and calculating their relationship, and going row by row or
 item by item.

This work was done for the Adaptive Rationality Center at the Max Planck Institute for Human Development
under the supervision of Dr. Dirk Wulff between the months of December 2023 and February 2024 by José Francisco Chonay.