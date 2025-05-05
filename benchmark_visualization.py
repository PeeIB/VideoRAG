'''
Questions asked:
    - When is token placement covered?
    - When is the fifteen puzzle game covered?
    - When is the k sat reconfiguration covered?
    - When is the graph coloring example given?
    - Why do we care about reconfiguration problems?
    - Where are token jumping and token sliding introduced?
    - Where are the technical details covered?
    - Where do they mention large language models?
    - Where do they cover details about quantum computing?
    - Where are p, np, and p space classes defined?
'''

import matplotlib.pyplot as plt
import numpy as np

# %%
scores = [0.7, 0.7, 0.7, 0.5, 0.4, 0.5]
labels = ['FAISS', 'IVFFLAT', 'HNSW', 'TF-IDF', 'BM25', 'Image']

# %% 
plt.bar(np.arange(6), scores, color = '#7038DD')
plt.xticks(np.arange(6), labels)
plt.xlabel('Model')
plt.ylabel('Score in %')
plt.title('Scores in percentage achieved \n by each model on the gold standard test set')
plt.savefig('GoldTest.pdf', bbox_inches='tight')
plt.show()