import logging
import azure.functions as func
import pandas as pd
import numpy as np
from azure.storage.blob import BlobServiceClient
import os

# Azure Blob Storage connection string
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def get_list_all_reco(userID, clicks, dico, n_reco=5):
    '''Return 5 recommended articles ID to user'''

    # Get the list of articles viewed by the user
    var = clicks.loc[clicks.user_id == userID]['article_id'].to_list()
    list_of_reco = []
    for article in var :
        article_ids = dico[article]
        for article_id in article_ids:
            if article_id in var:
                article_ids.remove(article_id)
        list_of_reco.extend(article_ids)
    
    # Convert the list to a pandas Series
    example_series = pd.Series(list_of_reco)

    # Get reco for the last article
    last_reco = list_of_reco[:n_reco]

    # Count the number of occurences of each
    value_counts = example_series.value_counts()
    filtered_value_counts = value_counts[value_counts > 1]

    if len(filtered_value_counts)  >= n_reco:
        # get the n_reco most common articles
        reco = filtered_value_counts.index[:n_reco].tolist()
    else :
        # get the n_reco most common articles and complete with the last reco
        reco = filtered_value_counts.index.tolist()
        last_reco = list_of_reco[:n_reco]
        i = 0
        while len(reco) < n_reco and i < len(last_reco):
            if last_reco[i] not in reco:
                reco.append(last_reco[i])
            i += 1

    return reco


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    user_id = req.params.get('userID')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('userID')

    if user_id:
        # Connect to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_name = "projet10"

        # Download the clicks and embeddings data from Blob Storage
        clicks_blob_client = blob_service_client.get_blob_client(container=container_name, blob="clicks_storage.csv")
        embeddings_blob_client = blob_service_client.get_blob_client(container=container_name, blob="reco_by_article.pkl")

        clicks = pd.read_csv(clicks_blob_client.download_blob().readall())

        with open(embeddings_blob_client.download_blob().readall(), 'rb') as f:
            embeddings = pd.read_pickle(f)

        # Get recommendations
        recommendations = get_list_all_reco(user_id, clicks, embeddings)

        return func.HttpResponse(f"Recommended articles: {recommendations}")
    else:
        return func.HttpResponse(
            "Please pass a userID on the query string or in the request body",
            status_code=400
        )