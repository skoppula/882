import numpy as np
import csv
import matplotlib.pyplot as plt
# pip install xlmhg
import xlmhg

def enrichment_mHGtest(vocab_file_name, ranked_genes_file_name, save_file_name = 'enrichment', n_sig_pathways = 6, enrichment_minimum_n = 5, lowest_cutoff = 100):
    vocab = np.load(vocab_file_name)

    gene_sets = []
    gene_set_sizes = []
    gene_set_names = []

    print('Loading gene sets...(can take a few minutes)')
    # load gene sets from msigdb file
    with open('c2.all.v6.0.symbols.gmt', 'r') as file:
        reader = csv.reader(file, delimiter = '\t')
        for idx, row in enumerate(reader):
            gene_set_names.append(row[0])
            genes = np.array(row[2:])
            genes = genes[np.in1d(genes,vocab)]
            gene_sets.append(genes)
            gene_set_sizes.append(len(genes))
    print('Done')
    gene_set_names = np.array(gene_set_names)
    gene_set_sizes = np.array(gene_set_sizes)
    gene_subsize = gene_set_sizes[gene_set_sizes>10]

    idx = np.argwhere(gene_set_sizes>10).squeeze()

    names = gene_set_names[idx]
    sets = [gene_sets[i] for i in idx]
    sizes = gene_set_sizes[idx]

    ranked_gene_lists = np.load(ranked_genes_file_name)

    all_pvals = []
    all_pathways = []

    print('Calculating enrichment scores')
    for idx, gene_list in enumerate(ranked_gene_lists):
        scores = []
        for msigdbset in sets:
            gene_in_set = np.in1d(gene_list, msigdbset)+0
            _,_,pval = xlmhg.xlmhg_test(gene_in_set, enrichment_minimum_n, lowest_cutoff)
            scores.append(pval)
        scores = np.array(scores)

        min_indices = np.argpartition(scores, n_sig_pathways)[0:n_sig_pathways]
        min_scores = scores[min_indices]
        best_pathways = names[min_indices]
        min_indices = np.argsort(min_scores)
        min_scores = min_scores[min_indices]
        best_pathways = best_pathways[min_indices]

        all_pvals.append(min_scores)
        all_pathways.append(best_pathways)
        if idx % 10 == 0:
            print(idx+1,'of', len(ranked_gene_lists))
    print('Done')

    print('Saving results')
    all_pvals = np.array(all_pvals)
    np.save(save_file_name+'_pvals.npy', all_pvals)

    with open(save_file_name+'_genes.txt','w') as file:
        writer = csv.writer(file, delimiter = ' ')
        for pathway in all_pathways:
            writer.writerow(pathway)
    print('Done')
    return all_pvals, all_pathways
