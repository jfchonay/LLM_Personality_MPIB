from sentence_transformers import SentenceTransformer
import pandas as pd
import os

if __name__ == "__main__":
    root = '/Users/josechonay/Library/CloudStorage/OneDrive-CarlvonOssietzkyUniversitätOldenburg/Winter Semester ' \
           '23-24/Internship/ARC'
    data_set = 'Open_Source_Psychometrics'
    data = 'items'

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    correlations = pd.read_csv((os.path.join(root, data_set, data) + '.tsv'), header=0)
    sentences = correlations['item'].tolist()
    embeddings = model.encode(sentences)
