from pathlib import Path
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from experiments.text_with_hook import opening_text
from kairos.data import load_labels, load_posts


DATA = Path('/Volumes/caelmbleidd/lab1/dataset/core')
OUTPUT = Path('/Users/ekaterina/kairos-dataset/results/embeddings/hooks_e5_small.npz')
MODEL_NAME = 'intfloat/multilingual-e5-small'
OPENING_DEFINITION = 'first_nonempty_line_max_300_characters'


def main():
    labels = load_labels(DATA)
    posts = load_posts(DATA)
    keys = labels.loc[labels['split'] == 'train', ['author', 'post']]
    selected = keys.merge(
        posts[['author', 'post', 'text']],
        on=['author', 'post'],
        how='left',
    )
    openings = selected['text'].map(opening_text)
    has_text = openings.ne('').to_numpy()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(MODEL_NAME, device=device)
    vectors = np.zeros((len(selected), 384), dtype=np.float32)
    started = time.perf_counter()

    if has_text.any():
        vectors[has_text] = model.encode(
            ['query: ' + text for text in openings[has_text]],
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    with temporary.open('wb') as destination:
        np.savez_compressed(
            destination,
            author=selected['author'].to_numpy(dtype=str),
            post=selected['post'].to_numpy(dtype=str),
            vectors=vectors,
            has_text=has_text,
            model_name=np.array(MODEL_NAME),
            prefix=np.array('query: '),
            max_seq_length=np.array(model.max_seq_length),
            opening_definition=np.array(OPENING_DEFINITION),
        )
    temporary.replace(OUTPUT)
    # print('shape', vectors.shape)
    # print(f'time - {(time.perf_counter() - started) / 60:.1f} min.')
    # print('saved', OUTPUT)


if __name__ == '__main__':
    main()
