
# coding: utf-8

# In[3]:

import pandas as pd
import os.path
import sys
import numpy as np


# In[26]:

def cut_out_zero_rows(data):
    return data.loc[~(data == 0).all(axis=1)]

def cut_out_zero_cols(data):
    return data.loc[:, ~(data == 0).all(axis=0)]

def load_dataset(preprocessed_path='dataset/preprocessed/human_melanoma_data.npy',
                 raw_path='dataset/human_melanoma_data.tsv', filter_std=None, load=True):

    preprocessed_data_path = preprocessed_path.split('.')[0] + '_' + str(filter_std) + '.npy'
    if os.path.isfile(preprocessed_data_path) and load:
        return np.load(preprocessed_data_path),  np.load(preprocessed_path.split('.')[0] + "_column_lbls.npy"), np.load(preprocessed_path.split('.')[0] + "_row_lbls.npy")

    elif os.path.isfile(raw_path):
        expr_data = pd.read_csv(raw_path, delimiter='\t').T
        print("Number of cells:", expr_data.shape[0])
        print("Number of genes:", expr_data.shape[1])

        # excise stop genes
        STOP_GENES = ['ACTB', 'B2M', 'EEF1A1', 'MTRNR2L1', 'MTRNR2L2',
                  'MTRNR2L8', 'RPL41', 'RPLP1', 'RPS18', 'RPS27', 'TMSB4X']
        expr_data = expr_data.drop(STOP_GENES, 1)
        print("After cutting out stop genes:")
        print("\tNumber of cells:", expr_data.shape[0])
        print("\tNumber of genes:", expr_data.shape[1])

        genes = list(expr_data.columns.values)
        ribosomal_genes = [gene for gene in genes if gene.startswith('RPL')]
        mito_ribosomal_genes = [gene for gene in genes if gene.startswith('MRPL')]
        excise_genes = ribosomal_genes + mito_ribosomal_genes
        expr_data = expr_data.drop(excise_genes, 1)
        print("After cutting out ribosomal protein genes:")
        print("\tNumber of cells:", expr_data.shape[0])
        print("\tNumber of genes:", expr_data.shape[1])

        # excise zero cells
        expr_data = cut_out_zero_rows(expr_data)
        print("After cutting out all-zero rows [cells]:")
        print("\tNumber of cells:", expr_data.shape[0])
        print("\tNumber of genes:", expr_data.shape[1])

        # excise zero genes
        expr_data = cut_out_zero_cols(expr_data)
        print("After cutting out all-zero columns [genes]:")
        print("\tNumber of cells:", expr_data.shape[0])
        print("\tNumber of genes:", expr_data.shape[1])
        
        # remove low STD cols (genes)
        stds = np.std(expr_data, axis=0)
        keep_idxs = stds.argsort()[int(len(stds)*filter_std):]
        threshold = np.sort(stds)[int(len(stds)*filter_std)]
        expr_data = np.take(expr_data,keep_idxs, axis=1)
        print("After cutting out the bottom %f fraction of genes with low STD (<%f):" % (filter_std, threshold))
        print("\tNumber of cells:", expr_data.shape[0])
        print("\tNumber of genes:", expr_data.shape[1])
        
        expr_data = expr_data.astype('int')

        if not os.path.isdir('dataset/preprocessed/'):
            os.mkdir('dataset/preprocessed/')

        np.save(preprocessed_data_path, expr_data.values)
        np.save(preprocessed_path.split('.')[0] + "_column_lbls.npy", expr_data.columns.values)
        np.save(preprocessed_path.split('.')[0] + "_row_lbls.npy", expr_data.index.values)

        return expr_data.values, expr_data.columns.values, expr_data.index.values

    else:
        print("Couldn't find files!")
        sys.exit(1)


# In[27]:

# data, genes, cells = load_dataset(filter_std=0.25)
# print("Data shape:", data.shape)


# In[28]:

# data, genes, cells = load_dataset(filter_std=0.1)
# print("Data shape:", data.shape)


# In[29]:

# data, genes, cells = load_dataset(filter_std=0)
# print("Data shape:", data.shape)

