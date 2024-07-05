import meilisearch
import json
from model import document_to_vector


def create_index():
    client = meilisearch.Client('http://localhost:7700')

    json_file = open('movies2.json', encoding='utf-8')
    movies = json.load(json_file)
    client.index('movies-vectorized').add_documents(movies)


def vectorize_documents():
    client = meilisearch.Client('http://localhost:7700')
    with open('movies2.json', encoding='utf-8') as json_file:
        movies = json.load(json_file)

    for movie in movies:
        print(movie["id"], movie["title"])
        vector = document_to_vector(
            f"{movie['title']} {movie['overview']}").tolist()
        client.index('movies-vectorized').update_documents([{
            'id': movie["id"],
            '_vectors': {"bart": vector}
        }])


if __name__ == "__main__":
    embeddings = {
        "bart": {
            "source": "userProvided",
            "dimensions": 768
        },
        "huggingFace": {
            "source": "huggingFace",
            "model": "BAAI/bge-base-en-v1.5",
            "documentTemplate":
                "{{doc.title}} {{doc.overview|truncatewords: 20}}"
        },
    }

    client = meilisearch.Client('http://localhost:7700')

    # create_index()
    # vectorize_documents()
    # client.index('movies-vectorized').update_filterable_attributes([
    #     'id',
    #     'genres',
    #     'release_date'
    #     ])
    # client.index('movies-vectorized').update_sortable_attributes([
    #     'title',
    #     'release_date'
    #     ])

    # client.index('movies').update_filterable_attributes([
    #     'id',
    #     'genres',
    #     'release_date'
    #     ])
    # client.index('movies').update_sortable_attributes([
    #     'title',
    #     'release_date'
    #     ])

    # client.index('movies-vectorized').update_embedders(embeddings)
    print(client.index('movies-vectorized').get_tasks().results[0])
    print(client.index('movies-vectorized').get_embedders())


# Enable vector embedding
# curl \
#   -X PATCH 'http://localhost:7700/experimental-features/' \
#   -H 'Content-Type: application/json'  \
#   --data-binary '{
#     "vectorStore": true
#   }'

# curl \
#   -X PATCH 'http://localhost:7700/indexes/movies/settings' \
#   -H 'Content-Type: application/json' \
#   --data-binary '{
#     "embedders": {
#       "default": {
#         "source": "huggingFace",
#         "model": "BAAI/bge-base-en-v1.5",
#         "documentTemplate": "A movie titled {{doc.title}} whose description starts with {{doc.overview|truncatewords: 20}}"
#       }
#     }
#   }'


# print(client.index('movies').search("meo", {"hybrid": {"semanticRatio": 1.0,
#                                                  "embedder": "default"}}))
# client.index('movies').reset_embedders()
# print(client.index('movies-vectorized').get_tasks().results[0])
# client.cancel_tasks({"uids": ["11"]})
# print(client.index('movies').search({"hybrid": {"semanticRatio": 1.0, "embedder": "bart"}, "vector": document_to_vector()}))
# print(client.index('movies').wait_for_task(11))
