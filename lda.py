
# coding: utf-8

# In[16]:

from preprocess_dataset import load_dataset
from preprocess_dataset import cut_out_zero_rows
from preprocess_dataset import cut_out_zero_cols
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
import os
import errno
import time


# In[17]:

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def compute_stats(data):
    tmp = np.count_nonzero(data, axis=1)
    print("Average/Min/Max non-zero count for each cell: ", np.mean(tmp), np.min(tmp), np.max(tmp))
    tmp = np.count_nonzero(data, axis=0)
    print("Average/Min/Max non-zero count for each gene: ", np.mean(tmp), np.min(tmp), np.max(tmp))
    if np.max(tmp) == data.shape[0]: print("there exists a gene active in all cells!")
    tmp = np.max(data, axis=0)
    print("Average/Min/Max max-expression value for each gene: ", np.mean(tmp), np.min(tmp), np.max(tmp))
    print("Number of genes with max-expression of 1:", np.where(tmp==1)[0].shape[0])
    # mean non-zero expression value of each gene
    tmp = np.true_divide(data.sum(0),(data!=0).sum(0))
    print("Average/Min/Max mean non-zero expression value for each gene: ", np.mean(tmp), np.min(tmp), np.max(tmp))
    
def normalize_along_columns(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def split_test(data, test_size=0.1):
    data_trn,data_te,_,_ = train_test_split(data, np.ones(data.shape[0]), test_size=test_size, random_state=42)
    return data_trn,data_te

def gene_dropout(data_trn,data_te,dropout=0.1):
    column_dropout = 0.1
    rand_idx = np.random.randint(data_trn.shape[1], size=int(data_trn.shape[1]*column_dropout))
    trimmed_data_trn = data_trn[:,rand_idx]
    trimmed_data_te = data_te[:,rand_idx]
    
    trimmed_data_te = cut_out_zero_rows(pd.DataFrame(trimmed_data_te)).as_matrix()
    keep_cols = ~(pd.DataFrame(trimmed_data_te) == 0).all(axis=0)
    trimmed_data_te = cut_out_zero_cols(pd.DataFrame(trimmed_data_te)).as_matrix()
    
    return trimmed_data_trn, trimmed_data_te

def save_model(model, folder = "", model_name="model"):
    folder = 'models/' + folder
    mkdir(folder)
    joblib.dump(model, folder + model_name + '.pickle')
    joblib.dump(model, folder + model_name + '.pickle')

def save_data(data, genes, path, as_txt=False):
    folder = "lda_data_bak/"
    mkdir(folder)
    np.save(folder + path + ".npy", data)
    if as_txt:
        with open(folder + path + ".txt", 'w') as f:
            for row in data:
                idxs = np.nonzero(row)
                f.write(" ".join(["%s %d" % t for t in zip(genes[idxs], row[idxs])]))
                f.write("\n")


# In[19]:

data, genes, cells = load_dataset(filter_std=0.1)
data_tr, data_te = split_test(data)
save_data(data_tr, genes, "train_data", as_txt=True)
save_data(data_te, genes, "test_data", as_txt=True)
print("train/test data shape:", data_tr.shape, data_te.shape)

run_ovb = True; run_gibbs = True
n_top_words = 15
print("# top words: %d" % n_top_words)


# ### sklearn online variational bayes (with mini-batches) 
# (hoffman, blei, et al. https://pdfs.semanticscholar.org/157a/ef34d39c85d6576028f29df1ea4c6480a979.pdf)

# In[ ]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_genes = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(("topic %d:" % topic_idx) + ", ".join(top_genes))


# In[13]:

if run_ovb:
    print("online variational bayes with sklearn")
    for n_clusters in [5,10,25,50]:
        t0 = time.time()
        print("number of clusters:", n_clusters)
        lda_trainer = LatentDirichletAllocation(n_clusters, n_jobs = 2, random_state=17)
        lda_trained = lda_trainer.fit(data_tr)
        print("\ttraining time:",time.time()-t0)
        save_model(lda_trained, folder='online_vb/', model_name=str(n_clusters))
        print_top_words(lda_trained, genes, n_top_words)
        log_likelihood = lda_trained.score(data_te)
        perplexity = lda_trained.perplexity(data_te)
        print("\tlog likelihood:", log_likelihood)
        print("\tperplexity:", perplexity)


# ### parallelized gibbs sampling
# (liu, et al.https://github.com/openbigdatagroup/plda)

# In[10]:

def print_top_words(topic_dists, feature_names, n_top_words):
    for topic_idx, topic in enumerate(topic_dists):
        top_genes = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(("topic %d:" % topic_idx) + ", ".join(top_genes))


# In[23]:

if run_gibbs:
    print("collapsed gibbs sampling with lda")

    cmd1 = "mpiexec -n 2 ./plda/mpi_lda --num_topics %d --alpha 0.1 --beta 0.01 --training_data_file %s "
    cmd2 = "--model_file %s --burn_in_iterations 100 --total_iterations 100"
    cmd = cmd1 + cmd2

    mkdir("results/gibbs/")

    for n_clusters in [5,10,25,50]:
        print("number of clusters:", n_clusters)
        final_model_path = "results/gibbs/%d_model.txt" % n_clusters
        final_cmd = cmd % (n_clusters, "lda_data_bak/train_data.txt", final_model_path)

        print("executing", final_cmd)
        t0 = time.time()
        os.system(final_cmd)
        print("\ttraining time:",time.time()-t0)

        model_frm = pd.read_csv(final_model_path, sep=r"\s+", header = None, index_col = 0)
        topic_dists = model_frm.as_matrix().T
        genes = list(model_frm.index)
        print_top_words(topic_dists, genes, n_top_words)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



