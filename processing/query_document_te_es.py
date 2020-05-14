import os
import time

import tensorflow.compat.v1 as tf
import tensorflow_hub as tf_hub

from elastic_client import ElasticClient, ESConnectionData


class QueryTEDocument:
    def __init__(self):
        self.GPU_LIMIT = 0.5
        self.google_api = "https://tfhub.dev/google/universal-sentence-encoder/2"
        self.embeddings = None
        self.session = None
        self.user_query = ''
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

    def query_definition(self, document_vector):
        query_structure = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['document_vector']) + 1.0",
                    "params": {"query_vector": document_vector}
                }
            }
        }

        search_start = time.time()
        response = self.connection_client.elastic_connection.search(
            index='smart_library_tf',
            body={
                "size": 5,
                "query": query_structure,
                "_source": {"includes": ["document_title", "document_content"]}
            }
        )
        search_time = time.time() - search_start

        print()
        print("{} total hits.".format(response["hits"]["total"]["value"]))
        print("search time: {:.2f} ms".format(search_time * 1000))
        for hit in response["hits"]["hits"]:
            print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
            print(hit["_source"])
            print()

    def query_document(self, document_path):
        document = open(document_path, "r")
        document_content = str(document.read())
        document_vector = self.embed_input_text([document_content])
        self.query_definition(document_vector=document_vector)

    def query_input(self):
        while True:
            try:
                self.user_query = input("Search for something: ")
                user_query_vector = self.embed_input_text([self.user_query])
                self.query_definition(user_query_vector)
            except KeyboardInterrupt:
                return


if __name__ == '__main__':
    query_es_te = QueryTEDocument()
    # query_es_te.query_document(document_path="D:\\Proiecte\\Smart-Library\\data\\document\\query_document.txt")
    query_es_te.query_input()
    query_es_te.stop_tensorflow_session()
