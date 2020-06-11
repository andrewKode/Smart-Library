import uuid
import os
import csv

import tensorflow.compat.v1 as tf
import tensorflow_hub as tf_hub
import pandas as pd

from elastic_client import ElasticClient, ESConnectionData


class IngestDocumentTextEmbeddings:
    def __init__(self):
        self.GPU_LIMIT = 0.5
        self.google_api = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.embeddings = None
        self.session = None
        self.docs_count = 0
        self.docs_limit = 0
        self.text_phrase = ''
        self.credentials = ESConnectionData().connection_credentials
        self.connection_client = ElasticClient(hosts=self.credentials['hosts'],
                                               user=self.credentials['user'],
                                               password=self.credentials['password'])
        if not self.connection_client.elastic_connection.ping():
            raise ValueError("Connection with the elastic database has failed.")
        self.initialize_tensorflow_session()

    def initialize_tensorflow_session(self):
        # disable CPU instructions support warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.prepare_embeddings()
        self.prepare_tensorflow_session()

    def stop_tensorflow_session(self):
        self.session.close()

    def prepare_embeddings(self):
        print("Loading pre-trained embeddings...")
        embed = tf_hub.Module(self.google_api)
        self.text_phrase = tf.placeholder(tf.string)
        self.embeddings = embed(self.text_phrase)

    def prepare_tensorflow_session(self):
        session_config = tf.ConfigProto()
        session_config.gpu_options.per_process_gpu_memory_fraction = self.GPU_LIMIT
        self.session = tf.Session(config=session_config)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def embed_input_text(self, text_data):
        # here is where the magic happens
        vectors = self.session.run(self.embeddings, feed_dict={self.text_phrase: text_data})
        return [vector.tolist() for vector in vectors][0]

    def index_definition(self, data, title):
        self.docs_count += 1
        document_title = title
        document_content = data
        document_vector = self.embed_input_text([document_content])
        document_id = uuid.uuid1().hex
        body_data = {'document_title': document_title,
                     'document_content': document_content,
                     'document_vector': document_vector}

        self.connection_client.elastic_connection.index(index="smart_library_tf", id=document_id, body=body_data)

    def ingest_from_csv_corpus(self, docs_limit, csv_path, content_column_name, title_column_name, csv_reader):
        self.docs_limit = docs_limit
        if csv_reader == "pandas":
            # THIS IS WITH PANDAS
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data.dropna().reset_index(drop=True)
            csv_data_content = csv_data[content_column_name]
            csv_data_title = csv_data[title_column_name]
            try:
                print("Indexing documents from csv file.")
                for data in csv_data_content:
                    self.index_definition(data, csv_data_title)
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
                    self.index_definition(csv_data_content, csv_data_title)
                    if self.docs_limit == self.docs_count:
                        print(f"Done indexing {self.docs_count} documents.")
                        break
                except TypeError:
                    pass
        else:
            print("Incorrect csv reading method mentioned, pls chose betwen \"pandas\" or \"python-csv\".")

    def index_document(self, document_path, title):
        document_title = title
        document = open(document_path, "r")
        document_content = str(document.read())
        self.index_definition(document_content, document_title)
        print("Done indexing.")


if __name__ == '__main__':
    print("Indexing document(s)...")
    ingest_doc_tf = IngestDocumentTextEmbeddings()
    # ingest_doc_tf.index_document(document_path="D:\\Proiecte\\Smart-Library\\data\\document\\ingest_document.txt",
    #                              title="Some random title.")
    ingest_doc_tf.ingest_from_csv_corpus(csv_path="D:\\Proiecte\\CorpusAndDataset\\articles1.csv",
                                         content_column_name="content",
                                         title_column_name="title",
                                         csv_reader="python-csv",
                                         docs_limit=5000)
    ingest_doc_tf.stop_tensorflow_session()
