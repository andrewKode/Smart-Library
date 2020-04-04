import pickle as pickle_save_model
import pandas as pd
import numpy as np

from tqdm import tqdm
from langdetect import detect
from itertools import chain
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models import Phrases

from store_lda_training_data import DataStorage, LDAModel


class LDATraining:
    """
    @author: Andrei Iancu
    @version: 1.0.0
    @description: This module is being used to train LDA models. We will be using this LDA model to process text
    documents which will be later ingested into a elastic search database.
    """
    def __init__(self):
        # init pandas()
        tqdm.pandas()
        self.training_data_available = False
        self.store_data = DataStorage()
        self.lda_model_utils = LDAModel()

    def read_training_data(self, training_data, column_name):
        if training_data.endswith(".csv"):
            processed_training_data = pd.read_csv(training_data)
            # remove any missing values
            processed_training_data = processed_training_data.dropna().reset_index(drop=True)
            processed_training_data = processed_training_data[column_name]
            try:
                for training_sample in processed_training_data:
                    if detect(training_sample) != 'en':
                        print("You are using training data which is not 100% english.\n")
            except TypeError:
                pass
            self.store_data.training_raw_data = processed_training_data
            print("\n...finished loading raw data")
            if not self.store_data.training_raw_data.empty:
                self.training_data_available = True
        else:
            print("The used training document should be a .csv file.\n")

    def process_training_sentences(self):
        processed_training_data = self.store_data.training_raw_data
        processed_training_data = processed_training_data.progress_map(sent_tokenize)
        self.store_data.training_sentences_data = processed_training_data
        print("\n...finished processing sentences")
        # print(self.store_data.training_sentences_data)

    def process_training_words(self):
        processed_training_data = self.store_data.training_sentences_data
        processed_training_data = processed_training_data.progress_map(lambda sentences: [word_tokenize(sentence)
                                                                                          for sentence in sentences])
        self.store_data.training_words_data = processed_training_data
        print("\n...finished processing words")
        # print(self.store_data.training_words_data)

    def process_pos_tag(self):
        processed_training_data = self.store_data.training_words_data
        processed_training_data = processed_training_data.progress_map(lambda tokens_sentences: [pos_tag(tokens)
                                                                                                 for tokens
                                                                                                 in tokens_sentences])
        self.store_data.training_pos_tag_data = processed_training_data
        print("\n...finished processing pos tag")
        # print(self.store_data.training_pos_tag_data)

    def process_words_lem(self):

        def get_word_pos_role(tree_bank_tag):

            if tree_bank_tag.startswith('J'):
                return wordnet.ADJ
            elif tree_bank_tag.startswith('V'):
                return wordnet.VERB
            elif tree_bank_tag.startswith('N'):
                return wordnet.NOUN
            elif tree_bank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''

        processed_training_data = self.store_data.training_pos_tag_data
        word_net_lem = WordNetLemmatizer()
        processed_training_data = processed_training_data.progress_map(
            lambda list_tokens_pos: [
                [
                    word_net_lem.lemmatize(el[0], get_word_pos_role(el[1]))
                    if get_word_pos_role(el[1]) != '' else el[0] for el in tokens_POS
                ]
                for tokens_POS in list_tokens_pos
            ]
        )
        self.store_data.training_lem_data = processed_training_data
        print("\n...finished processing lem words")
        # print(self.store_data.training_lem_data)

    def remove_stopwords(self):
        removable_stopwords = stopwords.words('English') + self.lda_model_utils.helper_stopwords_other + \
                              self.lda_model_utils.helper_stopwords_verbs

        processed_training_data = self.store_data.training_lem_data.map(lambda sentences:
                                                                        list(chain.from_iterable(sentences)))
        processed_training_data = processed_training_data.map(lambda tokens: [token.lower() for token in tokens
                                                                              if token.isalpha() and token.lower()
                                                                              not in removable_stopwords
                                                                              and len(token) > 1])
        self.store_data.lem_tokens = processed_training_data
        print("\n...finished removing the stopwords")
        # print(self.store_data.lem_tokens)

    def run_training(self):
        processed_tokens = self.store_data.lem_tokens.tolist()
        bi_gram_model = Phrases(processed_tokens)
        tri_gram_model = Phrases(bi_gram_model[processed_tokens], min_count=1)
        processed_tokens = list(tri_gram_model[bi_gram_model[processed_tokens]])

        lda_dictionary = corpora.Dictionary(processed_tokens)
        lda_dictionary.filter_extremes(no_below=3)
        corpus = [lda_dictionary.doc2bow(token) for token in processed_tokens]

        np.random.seed(123456)
        num_topics = 100
        lda_model = models.LdaModel(corpus, num_topics=num_topics,
                                    id2word=lda_dictionary,
                                    passes=4, alpha=[0.01] * num_topics,
                                    eta=[0.01] * len(lda_dictionary.keys()))
        store_model = open("D:\\Proiecte\\Smart-Library\\model\\lda.model", "wb")
        store_dictionary = open("D:\\Proiecte\\Smart-Library\\model\\lda_dict.dictionary", "wb")
        pickle_save_model.dump(lda_model, store_model)
        pickle_save_model.dump(lda_dictionary, store_dictionary)
        # TODO: Implement a way to re-train the model with new data, or at lease gather more data.


if __name__ == '__main__':
    lda_training = LDATraining()
    lda_training.read_training_data(training_data="D:\\Proiecte\\Smart-Library\\data\\bbc\\BBC_news_dataset.csv",
                                    column_name="description")

    lda_training.process_training_sentences()
    lda_training.process_training_words()
    lda_training.process_pos_tag()
    lda_training.process_words_lem()
    lda_training.remove_stopwords()
    lda_training.run_training()
