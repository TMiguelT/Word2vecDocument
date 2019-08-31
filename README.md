# doc2vec_agg

## Installation
```bash
pip install git+https://github.com/TMiguelT/doc2vec_agg.git
```

## Usage
```python
from word2vec_agg.word2vec import docvector

# Generate the document vector
doc_vector = docvector(
    word2vec='./GoogleNews-vectors-negative300.bin', # Path to pretrained doc2vec embeddings, in binary format
    text=['passenger', 'terminal', 'building'], # Array of preprocessed tokens, representing the document
    max=True, # True if you want the maximum of each dimension in the final output
    mean=True, # True if you want the mean of each dimension in the final output
    min=True # True if you want the minimum of each dimension in the final output
)

# Do operations with the vector
from scipy.spatial import distance

return distance.cosine(doc_vector_1, doc_vector_2)

```