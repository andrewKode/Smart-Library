import uuid
import csv

import pickle as pickle_read_model
import pandas as pd

from nltk.tokenize import word_tokenize
import elastic_client


class IngestDocumentLDA:
    """
    We will be using this class for processing new documents. All these documents will be stored in a es database
    using the elastic_client module.
    """
    def __init__(self):
        self.version = "1.0.0"
        self.document_content = ''
        self.docs_count = 0
        self.docs_limit = 0
        self.credentials = elastic_client.ESConnectionData().connection_credentials
        self.connection_client = elastic_client.ElasticClient(hosts=self.credentials['hosts'],
                                                              user=self.credentials['user'],
                                                              password=self.credentials['password'])
        if not self.connection_client.elastic_connection.ping():
            raise ValueError("Connection with the elastic database has failed.")

    def process_data_lda(self, **kwargs) -> dict:
        lda_model = kwargs['lda_model']
        lda_dictionary = kwargs['lda_dictionary']
        raw_data = kwargs['raw_data']

        self.document_content = raw_data

        lda_model_storage = open(lda_model, "rb")
        lda_dictionary_storage = open(lda_dictionary, "rb")
        lda_model = pickle_read_model.load(lda_model_storage)
        lda_dictionary = pickle_read_model.load(lda_dictionary_storage)

        tokens = word_tokenize(raw_data)
        data_to_view = pd.DataFrame(lda_model[lda_dictionary.doc2bow(tokens)])

        topics = []
        accuracies = []
        for data_type, info_array in data_to_view.iteritems():
            for info_element in info_array:
                if data_type == 0:
                    topics.append(info_element)
                else:
                    accuracies.append(info_element)
        topic_data = dict(zip(topics, accuracies))

        return topic_data

    def index_data(self, **kwargs):
        self.docs_count += 1
        lda_model = kwargs['lda_model']
        lda_dictionary = kwargs['lda_dictionary']
        raw_data = kwargs['raw_data']
        raw_data_title = kwargs['raw_data_title']

        topics = self.process_data_lda(lda_model=lda_model, lda_dictionary=lda_dictionary, raw_data=raw_data)
        body_data = {"title": raw_data_title,
                     "content": raw_data,
                     "topics_accuracies": topics,
                     "topics": list(topics.keys())}
        document_id = uuid.uuid1().hex
        self.connection_client.elastic_connection.index(index="smart_library_lda", id=document_id, body=body_data)

    def ingest_lda_data_from_csv(self, **kwargs):
        csv_reader = kwargs["csv_reader"]
        csv_path = kwargs["csv_path"]
        content_column_name = kwargs["content_column"]
        title_column_name = kwargs["title_column"]
        lda_model = kwargs['lda_model']
        lda_dictionary = kwargs['lda_dictionary']
        self.docs_limit = kwargs['docs_limit']

        if csv_reader == "pandas":
            # THIS IS WITH PANDAS
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data.dropna().reset_index(drop=True)
            csv_data_content = csv_data[content_column_name]
            csv_data_title = csv_data[title_column_name]
            try:
                print("Indexing documents from csv file.")
                for data in csv_data_content:
                    self.index_data(lda_model=lda_model, lda_dictionary=lda_dictionary,
                                    raw_data=data, raw_data_title=csv_data_title)
                    if self.docs_limit == self.docs_count:
                        print(f"Done indexing {self.docs_count} documents.")
                        break
            except TypeError:
                pass
        elif csv_reader == "python-csv":
            # THIS IS WITH CSV READER
            csv_fille = open(csv_path, encoding='utf8')
            reader = csv.DictReader(csv_fille, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            print("Indexing documents from csv file.")
            for row in reader:
                csv_data_title = row['title']
                csv_data_content = row['content']
                try:
                    self.index_data(lda_model=lda_model, lda_dictionary=lda_dictionary,
                                    raw_data=csv_data_content, raw_data_title=csv_data_title)
                    if self.docs_limit == self.docs_count:
                        print(f"Done indexing {self.docs_count} documents.")
                        break
                except TypeError:
                    pass
        else:
            print("Incorrect csv reading method mentioned, pls chose betwen \"pandas\" or \"python-csv\".")

    def extract_lda_data_txt(self, **kwargs):
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
        document_id = uuid.uuid1().hex

        print(book_data)
        self.connection_client.elastic_connection.index(index="smart_library_lda", id=document_id, body=book_data)


if __name__ == '__main__':
    print("Indexing document(s)...")
    ingest_document = IngestDocumentLDA()
    ingest_document.ingest_lda_data_from_csv(csv_path="/home/andrei/Proiecte/Smart-Library/data/articles1.csv",
                                             csv_reader="python-csv",
                                             lda_model="/home/andrei/Proiecte/Smart-Library/model/lda.model",
                                             lda_dictionary="/home/andrei/Proiecte/Smart-Library/model/lda_dict.dictionary",
                                             title_column="title",
                                             content_column="content",
                                             docs_limit=5000)
