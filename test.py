from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import whisperx
import torch
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8"  # Change to "int8" if low on GPU mem (may reduce accuracy)

# Load the WhisperX model
model = whisperx.load_model("base", DEVICE, compute_type=compute_type)

# Configure boto3 for DigitalOcean Spaces
DO_SPACES_ENDPOINT = os.getenv('DO_SPACES_ENDPOINT')
DO_SPACES_KEY = os.getenv('DO_SPACES_KEY')
DO_SPACES_SECRET = os.getenv('DO_SPACES_SECRET')
DO_SPACES_REGION = os.getenv('DO_SPACES_REGION')
DO_SPACES_BUCKET = os.getenv('DO_SPACES_BUCKET')

s3_client = boto3.client('s3',
                         region_name=DO_SPACES_REGION,
                         endpoint_url=DO_SPACES_ENDPOINT,
                         aws_access_key_id=DO_SPACES_KEY,
                         aws_secret_access_key=DO_SPACES_SECRET)

@app.post("/transcribe/")
async def transcribe_audio(file_key: str):
    try:
        # Download the file from DigitalOcean Space
        temp_file_location = f"/tmp/{file_key}"
        try:
            s3_client.download_file(DO_SPACES_BUCKET, file_key, temp_file_location)
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise HTTPException(status_code=403, detail="Credentials error")

        # Transcribe the audio file using WhisperX
        result = model.transcribe(temp_file_location)
        transcription = result['segments'][0]['text']

        # Remove the temporary file
        os.remove(temp_file_location)

        # Returns transcription
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
