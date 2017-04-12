import numpy as np
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
print(X.shape)

data = X[0:50,0:200]


import IBPCompoundDirichlet
import time

t_start = time.time()

model = IBPCompoundDirichlet.IBPCompoundDirichlet()
model.fit_data(data, 500, 5, 1, 0.1)

t_end = time.time()

print(t_end-t_start)

model.write_to_file('reuters_100docs_2000words_results')
