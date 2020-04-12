from elasticsearch import Elasticsearch


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class ElasticClient:
    def __init__(self, **kwargs):
        self.host = kwargs['hosts']
        self.user = kwargs['user']
        self.password = kwargs['password']
        self.elastic_connection = Elasticsearch(hosts=self.host, http_auth=(self.user, self.password),
                                                sniff_on_start=True, request_timeout=30)


@singleton
class ESConnectionData:
    def __init__(self):
        # getting read errors with this url
        # TODO: Fix the connection with the UPT elastic server.
        # self.connection_credentials = {"hosts": ["http://elastic.cm.upt.ro:9200/"],
        #                                "user": "andrei",
        #                                "password": "not_defined"}

        # development
        self.connection_credentials = {"hosts": ["http://localhost:9200/"],
                                       "user": "andrei",
                                       "password": "not_defined"}