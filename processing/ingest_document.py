import pickle as pickle_read_model
import pandas as pd

from nltk.tokenize import word_tokenize
from processing import elastic_client


class IngestDocument:
    """
    We will be using this class for processing new documents. All these documents will be stored in a es database
    using the elastic_client module.
    """
    def __init__(self):
        self.version = "1.0.0"
        self.document_content = ''
        self.credentials = elastic_client.ESConnectionData().connection_credentials
        self.connection_client = elastic_client.ElasticClient(hosts=self.credentials['hosts'],
                                                              user=self.credentials['user'],
                                                              password=self.credentials['password'])
        if not self.connection_client.elastic_connection.ping():
            raise ValueError("Connection with the elastic database has failed.")

    def extract_lda_data(self, **kwargs):
        lda_model = kwargs['lda_model']
        document_path = kwargs['document_path']
        lda_dictionary = kwargs['lda_dictionary']

        document = open(document_path, "rb")
        self.document_content = str(document.read())

        lda_model_storage = open(lda_model, "rb")
        lda_dictionary_storage = open(lda_dictionary, "rb")
        lda_model = pickle_read_model.load(lda_model_storage)
        lda_dictionary = pickle_read_model.load(lda_dictionary_storage)

        tokens = word_tokenize(self.document_content)
        data_to_view = pd.DataFrame(lda_model[lda_dictionary.doc2bow(tokens)])
        print(data_to_view)

        topics = []
        accuracies = []
        for data_type, info_array in data_to_view.iteritems():
            for info_element in info_array:
                if data_type == 0:
                    topics.append(info_element)
                else:
                    accuracies.append(info_element)
        topic_data = dict(zip(topics, accuracies))
        book_data = {"title": "First book I have in the database",
                     "topics": topic_data}

        print(book_data)
        self.connection_client.elastic_connection.index(index="smart_library", id="first_book", body=book_data)


if __name__ == '__main__':
    ingest_document = IngestDocument()
    ingest_document.extract_lda_data(document_path="D:\\Proiecte\\Smart-Library\\data\\document\\test_document.txt",
                                     lda_model="D:\\Proiecte\\Smart-Library\\model\\lda.model",
                                     lda_dictionary="D:\\Proiecte\\Smart-Library\\model\\lda_dict.dictionary")
