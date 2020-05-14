# Smart-Library

@author: Andrei Iancu
credits: ElasticSearch documentation (ElasticSearch community and maintainers).

This software platform uses ElasticSearch's powerfull search engines in order to process and compare text documents.

There are two main methods of processing text data: LDA and TextEmbeddings.

For the LDA (Latent Dirichlet Allocation), we are using a locally trained model on BBC news articles data.
The text embeddings is using tensorflow pre-trained embeddings downloaded from the hub (Google API).

The ElasticSearch index settings are present in cluster/ folder for each processing type.

The pre-trained LDA models present in the models/ folder cannot be decoded with pickle in a different machine environment, so in case
an error pops up, a new model has to be trained.

