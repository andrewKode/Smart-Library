import time

import pickle as pickle_read_model
import pandas as pd

from nltk.tokenize import word_tokenize
from elastic_client import ElasticClient, ESConnectionData


class QueryLDADocument:
    def __init__(self):
        self.user_query = ''
        self.credentials = ESConnectionData().connection_credentials
        self.connection_client = ElasticClient(hosts=self.credentials['hosts'],
                                               user=self.credentials['user'],
                                               password=self.credentials['password'])
        if not self.connection_client.elastic_connection.ping():
            raise ValueError("Connection with the elastic database has failed.")

    def process_data_lda(self, **kwargs) -> dict:
        lda_model = kwargs['lda_model']
        lda_dictionary = kwargs['lda_dictionary']
        raw_data = kwargs['raw_data']

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

    def query_data(self, **kwargs):
        topics = kwargs['query_topics']
        list_query_topics = []

        def build_query_body():
            for topic in topics:
                query_element = {"match": {"topics": topic}}
                list_query_topics.append(query_element)
            body = {
                "query": {"bool": {"should": list_query_topics}}}
            return body

        print(build_query_body())
        search_start = time.time()
        response = self.connection_client.elastic_connection.search(index='smart_library_lda', body=build_query_body())
        search_time = time.time() - search_start

        print()
        print("{} total hits.".format(response["hits"]["total"]["value"]))
        print("search time: {:.2f} ms".format(search_time * 1000))
        for hit in response["hits"]["hits"]:
            print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
            print(hit["_source"])
            print()

    def query_input(self, **kwargs):
        lda_model = kwargs['lda_model']
        lda_dictionary = kwargs['lda_dictionary']
        while True:
            try:
                self.user_query = input("Search for something: ")
                generated_topics = list(self.process_data_lda(raw_data=self.user_query,
                                                              lda_model=lda_model,
                                                              lda_dictionary=lda_dictionary).keys())
                self.query_data(query_topics=generated_topics)
            except KeyboardInterrupt:
                return


if __name__ == '__main__':
    query_lda = QueryLDADocument()
    query_lda.query_input(lda_model="D:\\Proiecte\\Smart-Library\\model\\lda.model",
                          lda_dictionary="D:\\Proiecte\\Smart-Library\\model\\lda_dict.dictionary")
