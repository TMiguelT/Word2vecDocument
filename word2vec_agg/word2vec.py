import os
from gensim.models import KeyedVectors
import typing
import numpy


def docvector(word2vec: typing.Union[os.PathLike, KeyedVectors], text: typing.Iterable[str], min=True, max=True,
              mean=True):

    # Read the embeddings if we don't have them already
    if isinstance(word2vec, os.PathLike) or isinstance(word2vec, str):
        word2vec = KeyedVectors.load_word2vec_format(os.fspath(word2vec), binary=True)

    # Pick out vectors for each relevant word
    indices = [word2vec.vocab[word].index for word in text if word in word2vec.vocab]
    array = word2vec.vectors[indices]

    # Build up a list of vectors
    arrs = []

    # Add each aggregate vector as requested
    if min:
        arrs.append(numpy.amin(array, 0))
    if max:
        arrs.append(numpy.amax(array, 0))
    if mean:
        arrs.append(numpy.mean(array, 0))

    return numpy.concatenate(arrs)
