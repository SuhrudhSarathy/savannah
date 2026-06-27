from google.cloud import storage


def upload_file_to_gcs(local_path: str, bucket_name: str, blob_path: str) -> str:
    """Uploads local_path to gs://bucket_name/blob_path. Returns the gs:// URI.

    Auth via Application Default Credentials (google.cloud.storage.Client()).
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_path).upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_path}"
