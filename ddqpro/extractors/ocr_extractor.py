import os
import time
import json
import boto3
import logging
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

from .base import BaseExtractor, ExtractionResult, UnsupportedFormatError

logger = logging.getLogger(__name__)

# Load AWS credentials from .env
load_dotenv()


class TextractExtractor(BaseExtractor):
    """
    AWS Textract-based extractor for PDFs.
    This extractor uploads the PDF to S3 (if not already there),
    starts a Textract job, polls until completion, and then extracts the text.
    """

    def __init__(self, region_name: str = None, bucket_name: str = None, quality_threshold: float = 0.5):
        super().__init__()
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.quality_threshold = quality_threshold
        # Create boto3 clients using the region specified
        self.s3_client = boto3.client("s3", region_name=self.region_name)
        self.textract_client = boto3.client("textract", region_name=self.region_name)
        logger.debug(f"{self.name} initialized with region {self.region_name} and bucket {self.bucket_name}")

    def _upload_to_s3(self, local_file: str, s3_key: str) -> None:
        """
        Uploads the file to S3 if it doesn't already exist.
        """
        try:
            existing_files = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_key)
            if "Contents" in existing_files:
                logger.info(f"File already exists in S3: s3://{self.bucket_name}/{s3_key}")
                return
            logger.info(f"Uploading {local_file} to s3://{self.bucket_name}/{s3_key}...")
            self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
            logger.info("Upload completed successfully.")
        except NoCredentialsError:
            raise NoCredentialsError("AWS credentials not found. Please configure using .env and load_dotenv().")
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    def _start_textract_job(self, s3_key: str) -> str:
        """
        Starts a Textract job for the PDF in S3 and returns the JobId.
        """
        try:
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={"S3Object": {"Bucket": self.bucket_name, "Name": s3_key}}
            )
            job_id = response["JobId"]
            logger.info(f"Textract job started with Job ID: {job_id}")
            return job_id
        except ClientError as e:
            raise Exception(f"AWS Textract job failed to start: {e}")

    def _wait_for_job(self, job_id: str, max_retries: int = 30, wait_seconds: int = 5) -> Dict:
        """
        Polls Textract for job completion and returns the result when job succeeds.
        """
        for attempt in range(max_retries):
            response = self.textract_client.get_document_text_detection(JobId=job_id)
            status = response.get("JobStatus", "")
            if status in ["SUCCEEDED", "FAILED"]:
                logger.info(f"Job {job_id} completed with status: {status}")
                return response
            logger.debug(f"Job {job_id} status: {status}. Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
        raise TimeoutError(f"Textract job {job_id} timed out after {max_retries * wait_seconds} seconds")

    def _parse_textract_response(self, response: Dict) -> str:
        """
        Parses the Textract response and extracts text from LINE blocks.
        """
        blocks = response.get("Blocks", [])
        lines = [block["Text"] for block in blocks if block.get("BlockType") == "LINE" and "Text" in block]
        return "\n".join(lines)

    @BaseExtractor._measure_extraction_time()
    def extract(self, file_path: str) -> ExtractionResult:
        """
        Implements text extraction using AWS Textract.
        """
        try:
            path = self._validate_file(file_path)
            if path.suffix.lower() != ".pdf":
                raise UnsupportedFormatError(f"TextractExtractor supports only PDF files, got: {path.suffix}")

            logger.info(f"Starting Textract extraction for: {file_path}")

            # Define S3 key based on file name
            s3_key = f"documents/{path.name}"
            # Upload the file to S3 if necessary
            self._upload_to_s3(file_path, s3_key)

            # Start Textract job
            job_id = self._start_textract_job(s3_key)
            # Wait for job completion
            response = self._wait_for_job(job_id)
            if response.get("JobStatus") != "SUCCEEDED":
                raise Exception(f"Textract job failed with status: {response.get('JobStatus')}")

            extracted_text = self._parse_textract_response(response)
            quality_score = self.evaluate_quality(extracted_text)

            logger.info(f"Textract extraction complete. Quality score: {quality_score:.2f}")
            return ExtractionResult(
                content=extracted_text,
                quality_score=quality_score,
                extractor_used=self.name,
                extraction_time=0.0,  # Set by decorator
                metadata={
                    "s3_key": s3_key,
                    "job_id": job_id,
                    "region": self.region_name,
                }
            )
        except Exception as e:
            logger.error(f"Textract extraction failed: {str(e)}")
            return ExtractionResult(
                content="",
                quality_score=0.0,
                extractor_used=self.name,
                error=str(e),
                success=False,
                extraction_time=0.0,
                metadata={"file_path": file_path}
            )

    def evaluate_quality(self, text: str) -> float:
        """
        A simple quality evaluation based on text length.
        More sophisticated metrics can be added later.
        """
        if not text:
            return 0.0
        # For now, quality is proportional to length; adjust as needed.
        length = len(text)
        # Assume that a quality text should be at least 100 characters and 1000 is ideal.
        if length < 100:
            return 0.0
        elif length >= 1000:
            return 1.0
        else:
            return (length - 100) / 900
