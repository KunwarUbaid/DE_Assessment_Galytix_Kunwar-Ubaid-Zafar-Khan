# File: word_processor.py
#Task 3: Making functions using OOP principles
import gensim
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PhraseSimilarity:
    def __init__(self, vectors_file_path, phrases_csv_path):
        self.word_vectors = self.load_word_vectors(vectors_file_path)
        self.phrases_df = pd.read_csv(phrases_csv_path, encoding='latin1')

    def load_word_vectors(self, file_path):
        wv = KeyedVectors.load_word2vec_format(file_path, binary=True, limit=1000000)
        wv.save_word2vec_format('vectors.txt', binary=False)
        return KeyedVectors.load_word2vec_format('vectors.txt', binary=False)

    def phrase_vector(self, phrase):
        words = phrase.split()
        vector_sum = np.zeros((self.word_vectors.vector_size,), dtype="float32")
        word_count = 0
        for word in words:
            if word in self.word_vectors:
                vector_sum = np.add(vector_sum, self.word_vectors[word])
                word_count += 1
        if word_count == 0:
            return None
        return vector_sum / word_count

    def calculate_similarity(self, phrase1, phrase2):
        vec1 = self.phrase_vector(phrase1)
        vec2 = self.phrase_vector(phrase2)
        if vec1 is None or vec2 is None:
            return None
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))

    def compute_similarities(self):
        similarities = []
        for idx, row in self.phrases_df.iterrows():
            phrase = row['Phrases']
            for inner_idx, inner_row in self.phrases_df.iterrows():
                inner_phrase = inner_row['Phrases']
                similarity = self.calculate_similarity(phrase, inner_phrase)
                similarities.append({
                    'Phrase 1': phrase,
                    'Phrase 2': inner_phrase,
                    'Similarity': similarity[0][0] if similarity is not None else None
                })
        return pd.DataFrame(similarities)

    def closest_match(self, input_phrase):
        max_similarity = -1
        closest_phrase = None
        for idx, row in self.phrases_df.iterrows():
            phrase = row['Phrases']
            similarity = self.calculate_similarity(input_phrase, phrase)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
                closest_phrase = phrase
        return closest_phrase, max_similarity
