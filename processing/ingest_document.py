import pickle as pickle_read_model
import pandas as pd

from nltk.tokenize import word_tokenize


class IngestDocument:
    """
    We will be using this class for processing new documents. All these documents will be stored in a es database
    using the elastic_client module.
    """
    def __init__(self):
        self.version = "1.0.0"
        self.document_content = ''

    def process_new_document(self, document_path, lda_model, lda_dictionary):
        document = open(document_path, "rb")
        self.document_content = str(document.read())

        lda_model_storage = open(lda_model, "rb")
        lda_dictionary_storage = open(lda_dictionary, "rb")
        lda_model = pickle_read_model.load(lda_model_storage)
        lda_dictionary = pickle_read_model.load(lda_dictionary_storage)

        tokens = word_tokenize(self.document_content)
        data_to_view = pd.DataFrame(lda_model[lda_dictionary.doc2bow(tokens)])

        # TODO: Define the results of the process in a JSON object
        for topic in data_to_view[0]:
            print(topic)
        for accuracy in data_to_view[1]:
            print(accuracy)


if __name__ == '__main__':
    ingest_document = IngestDocument()
    ingest_document.process_new_document(document_path="D:\\Proiecte\\Smart-Library\\data\\document\\test_document.txt",
                                         lda_model="D:\\Proiecte\\Smart-Library\\model\\lda.model",
                                         lda_dictionary="D:\\Proiecte\\Smart-Library\\model\\lda_dict.dictionary")
