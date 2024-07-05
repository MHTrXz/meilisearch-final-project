import uvicorn
import time
import meilisearch

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from model import document_to_vector


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/search")
async def search(query: str, index: str, search_type: Optional[float],
                 apply_semantic_on_keyword: Optional[str],
                 sort_order: str = None, sort_by: str = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 genres: Optional[str] = None,
                 model: Optional[str] = None):
    client = meilisearch.Client('http://localhost:7700')
    filters = []
    semantic_ratio = search_type / 100
    search_index = "movies-vectorized"
    embedder = "bert"
    sort = []

    if index == "movies":
        search_index = "movies"
    elif index == "movies-vectorized":
        search_index = "movies-vectorized"

    if genres is not None:
        temp = []
        for g in genres.split(','):
            temp.append("genres = " + g)
        res = "(" + " OR ".join(temp) + ")"
        filters.append(res)

    if start_date is not None and end_date is not None:
        start_date_unix = int(time.mktime(
            datetime.strptime(start_date, "%Y,%m,%d").timetuple()))
        end_date_unix = int(time.mktime(
            datetime.strptime(end_date, "%Y,%m,%d").timetuple()))
        filters.append(f"release_date {start_date_unix} TO {end_date_unix}")
    elif end_date is not None:
        end_date_unix = int(time.mktime(
            datetime.strptime(end_date, "%Y,%m,%d").timetuple()))
        filters.append(f"release_date < {end_date_unix}")
    elif start_date is not None:
        start_date_unix = int(time.mktime(
            datetime.strptime(start_date, "%Y,%m,%d").timetuple()))
        filters.append(f"release_date > {start_date_unix}")

    if sort_by is not None:
        if sort_order == "asc" and sort_by == "name":
            sort = ["title:asc"]
        elif sort_order == "desc" and sort_by == "name":
            sort = ["title:desc"]
        elif sort_order == "asc" and sort_by == "time":
            sort = ["release_date:asc"]
        elif sort_order == "desc" and sort_by == "time":
            sort = ["release_date:desc"]

    if model == "huggingFace":
        embedder = "huggingFace"
        if search_index == "movies":
            embedder = "default"
        if apply_semantic_on_keyword == "true":
            response = (client.index(search_index).search(query, {
                "filter": ' AND '.join(filters),
                "sort": sort
            }))
            ids = []
            for hit in response["hits"]:
                ids.append(f"id = {hit['id']}")

            response_filtered = (client.index(search_index).search(query, {
                "filter": ' OR '.join(ids),
                "hybrid": {
                    "semanticRatio": semantic_ratio,
                    "embedder": embedder,
                },
                "sort": sort
            }))
            return response_filtered

        response = (client.index(search_index).search(query, {
            "hybrid": {
                "semanticRatio": semantic_ratio,
                "embedder": embedder,
            },
            "filter": ' AND '.join(filters),
            "sort": sort
        }))
    elif model == "bert":
        embedder = "bart"
        vector = document_to_vector(query).tolist()
        if apply_semantic_on_keyword == "true":
            response = (client.index(search_index).search(query, {
                "filter": ' AND '.join(filters),
                "sort": sort
            }))
            ids = []
            for hit in response["hits"]:
                ids.append(f"id = {hit['id']}")

            response_filtered = (client.index(search_index).search(query, {
                "filter": ' OR '.join(ids),
                "hybrid": {
                    "semanticRatio": semantic_ratio,
                    "embedder": embedder,
                },
                "sort": sort,
                "vector": vector
            }))
            return response_filtered

        response = (client.index(search_index).search(query, {
            "hybrid": {
                "semanticRatio": semantic_ratio,
                "embedder": embedder,
            },
            "filter": ' AND '.join(filters),
            "sort": sort,
            "vector": vector
        }))
    else:
        response = (client.index(search_index).search(query, {
            "filter": ' AND '.join(filters),
            "sort": sort
        }))

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
