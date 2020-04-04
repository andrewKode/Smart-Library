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
    # TODO: Finish the implementation of the ElasticClient
    def __init__(self, host):
        self.host = host

    def connect(self):
        elastic_connection = Elasticsearch(hosts=self.host)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
