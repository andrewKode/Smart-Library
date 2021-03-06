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
        # self.connection_credentials = {"hosts": ["https://elastic.cm.upt.ro:9200/"],
        #                                "user": "admin",
        #                                "password": "admin"}

        # development
        self.connection_credentials = {"hosts": ["http://192.168.1.9:9200/"],
                                       "user": "andrei",
                                       "password": "not_defined"}
