def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


@singleton
class DataStorage:
    def __init__(self):
        self.training_raw_data = None
        self.training_sentences_data = None
        self.training_words_data = None
        self.training_pos_tag_data = None
        self.training_lem_data = None
        self.lem_tokens = None


@singleton
class LDAModel:
    def __init__(self):
        self.helper_stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see',
                                       'want', 'come', 'take', 'use', 'would', 'can']
        self.helper_stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also',
                                       'copyright', 'something']
