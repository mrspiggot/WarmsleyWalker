# ddqpro/extractors/textract_extractor.py

from abc import ABC, abstractmethod
from typing import Dict, Optional
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
import time

logger = logging.getLogger(__name__)


class DocumentExtractor(ABC):
    """Abstract base class for document extraction"""

    @abstractmethod
    def extract(self, file_path: str) -> Dict:
        """Extract content from document"""
        pass


class AWSTextractExtractor(DocumentExtractor):
    """AWS Textract implementation of document extraction"""

    def __init__(self, region_name: str = "us-east-1", bucket_name: Optional[str] = None):
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region_name)
        self.textract_client = boto3.client(
            "textract",
            region_name=region_name,
            endpoint_url=f"https://textract.{region_name}.amazonaws.com"
        )

    def extract(self, file_path: str) -> Dict:
        """Extract text from document using AWS Textract"""
        try:
            # Upload to S3 if bucket provided
            if self.bucket_name:
                s3_key = self._upload_to_s3(file_path)
                text = self._process_with_textract(s3_key)
            else:
                # Direct processing for small files
                with open(file_path, 'rb') as document:
                    text = self._process_local_file(document.read())

            return {
                'raw_content': text,
                'format': 'text',
                'success': True
            }

        except Exception as e:
            logger.error(f"Textract extraction failed: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

    def _upload_to_s3(self, file_path: str) -> str:
        """Upload file to S3 and return key"""
        if not self.bucket_name:
            raise ValueError("No S3 bucket configured")

        file_name = Path(file_path).name
        s3_key = f"documents/{file_name}"

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
            return s3_key

        except NoCredentialsError as e:
            logger.error("AWS credentials not found")
            raise

    def _process_local_file(self, file_bytes: bytes) -> str:
        """Process a local file directly with Textract"""
        try:
            response = self.textract_client.detect_document_text(
                Document={'Bytes': file_bytes}
            )
            return self._extract_text_from_response(response)
        except ClientError as e:
            logger.error(f"Textract processing failed for local file: {str(e)}")
            raise

    def _process_with_textract(self, s3_key: str) -> str:
        """Process an S3-stored file with Textract"""
        try:
            # Start async job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': self.bucket_name,
                        'Name': s3_key
                    }
                }
            )
            job_id = response['JobId']
            logger.info(f"Started Textract job {job_id} for {s3_key}")

            # Poll for completion
            result = self._wait_for_job_completion(job_id)
            return self._extract_text_from_response(result)

        except ClientError as e:
            logger.error(f"Textract processing failed for S3 file: {str(e)}")
            raise

    def _wait_for_job_completion(self, job_id: str, max_retries: int = 30) -> Dict:
        """Wait for Textract job completion with timeout"""
        for attempt in range(max_retries):
            try:
                response = self.textract_client.get_document_text_detection(JobId=job_id)
                status = response['JobStatus']

                if status == 'SUCCEEDED':
                    logger.info(f"Textract job {job_id} completed successfully")
                    return response
                elif status == 'FAILED':
                    error = response.get('StatusMessage', 'Unknown error')
                    logger.error(f"Textract job {job_id} failed: {error}")
                    raise Exception(f"Textract job failed: {error}")

                logger.debug(f"Job {job_id} status: {status}. Waiting...")
                time.sleep(5)

            except ClientError as e:
                logger.error(f"Error checking job status: {str(e)}")
                raise

        raise TimeoutError(f"Textract job {job_id} timed out after {max_retries} attempts")

    def _extract_text_from_response(self, response: Dict) -> str:
        """Extract text blocks from Textract response"""
        text_blocks = []

        # Handle both synchronous and asynchronous response formats
        blocks = response.get('Blocks', [])
        if not blocks and 'Blocks' in response.get('Blocks', {}):
            blocks = response['Blocks']['Blocks']

        for block in blocks:
            if block['BlockType'] == 'LINE':
                text_blocks.append(block['Text'])

        return '\n'.join(text_blocks)

    def cleanup(self, s3_key: str) -> None:
        """Clean up S3 resources after processing"""
        if self.bucket_name:
            try:
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                logger.info(f"Cleaned up s3://{self.bucket_name}/{s3_key}")
            except ClientError as e:
                logger.warning(f"Failed to cleanup S3 object: {str(e)}")


class DocumentExtractorFactory:
    """Factory for creating document extractors"""

    @staticmethod
    def create_extractor(extractor_type: str, **kwargs) -> DocumentExtractor:
        """Create an extractor instance based on type"""
        extractors = {
            'textract': AWSTextractExtractor,
            # Add other extractors here as needed
        }

        if extractor_type not in extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

        return extractors[extractor_type](**kwargs)