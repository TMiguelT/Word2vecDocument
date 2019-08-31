"""
Note: requires the WORD2VEC environment variable pointing at the .bin file
"""
import os
from gensim.models import KeyedVectors
from word2vec_agg.word2vec import docvector
import pytest
from gensim.utils import simple_tokenize
from scipy.spatial import distance

word2vec_path = os.environ['WORD2VEC']
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


@pytest.mark.parametrize(['start', 'close', 'far'],
                         [
                             ['the lack of', 'the absence of a', 'the question'],
                             ['summary of the', 'brief overview of the', 'try to'],
                             ['to resolve', 'solving', 'provision'],
                         ]
                         )
def test_closer_to(start, close, far):
    start = docvector(word2vec, simple_tokenize(start))
    close = docvector(word2vec, simple_tokenize(close))
    far = docvector(word2vec, simple_tokenize(far))

    close_dist = distance.cosine(start, close)
    far_dist = distance.cosine(start, far)

    assert close_dist < far_dist
