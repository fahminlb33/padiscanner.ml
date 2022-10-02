import os
import logging
import tempfile

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import get_current_username, get_settings
from app.predictor import PredictorService

settings = get_settings()
router = APIRouter(
    prefix="/analysis",
    tags=["Image Analysis"],
    responses={404: {"description": "Not found"}},
)

# create client
blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
container_client: ContainerClient = blob_service_client.get_container_client(settings.azure_storage_container_name)

# load model
predictor_service = PredictorService()
predictor_service.load_model(settings.model_path, settings.class_name_path)

logger = logging.getLogger(__name__)

class ImageModel(BaseModel):
    user_id: str
    prediction_id: str
    original_filename: str

@router.post("/image")
async def get_graph(model: ImageModel, _: str = Depends(get_current_username)):
    # load image from blob storage
    local_filepath = os.path.join(tempfile.gettempdir(), model.original_filename)
    with open(local_filepath, "wb") as f:
        original_blob_name = f"{model.user_id}/{model.prediction_id}/{model.original_filename}"
        blob_client: BlobClient = container_client.get_blob_client(original_blob_name)
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())

    # get file size
    file_size = os.path.getsize(local_filepath)
    logger.info(f"Uploaded file size: {file_size}")

    # resize image to maximum of MAX_HEIGHT
    logger.info(f"Constraining uploaded image size...")
    resized_path = predictor_service.constrain_image_size(local_filepath)

    # make prediction
    logger.info(f"Running prediction...")
    (prediction_proba, heatmap_path, superimposed_path, masked_path) \
        = predictor_service.predict(resized_path, tempfile.gettempdir())

    # upload to blob storage
    logger.info(f"Uploading results to blob storage...")
    blob_ext = os.path.splitext(model.original_filename)[1]
    
    # build response
    response_dict = {
        "predicted_class": predictor_service.get_most_likely_class(prediction_proba),
        "class_probabilities": {
            predictor_service.get_class_from_prediction(i): round(v, 4) for i, v in enumerate(prediction_proba.tolist())
        }
    }

    # upload queue
    upload_queue = [
        ("heatmap", heatmap_path, f"{model.user_id}/{model.prediction_id}/heatmap{blob_ext}"),
        ("superimposed", superimposed_path, f"{model.user_id}/{model.prediction_id}/superimposed{blob_ext}"),
        ("masked", masked_path, f"{model.user_id}/{model.prediction_id}/masked{blob_ext}"),
    ]

    # upload to blob storage
    for (key, path, blob_name) in upload_queue:
        # get blob client
        heatmap_blob_client = container_client.get_blob_client(blob_name)

        # open read stream
        with open(path, "rb") as data:
            # upload
            heatmap_blob_client.upload_blob(data)

        # set to response dict
        response_dict[key] = heatmap_blob_client.url

    # return data
    return response_dict