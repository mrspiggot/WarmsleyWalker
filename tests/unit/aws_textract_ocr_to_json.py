# script_aws_textract_ocr_to_json.py
#     pdf_file = "../../data/input/mw/28.03.2024 Factsheet Lumyna - MW TOPS UCITS Fund GBP B (acc).pdf"

# File: aws_textract_ocr_to_json.py

import os
import boto3
import json
import time
from botocore.exceptions import NoCredentialsError, ClientError
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# AWS Configuration
# BUCKET_NAME = "lucidate-ocr-text-extraction"
BUCKET_NAME = "lucidate-ocr-text-extraction-us-east-1"

REGION_NAME = "eu-north-1"  # Set to match your S3 bucket's region
LOCAL_PDF_PATH = "../../data/input/mw/28.03.2024 Factsheet Lumyna - MW TOPS UCITS Fund GBP B (acc).pdf"
S3_FILE_KEY = "documents/28.03.2024_Factsheet_Lumyna.pdf"  # Adjust path if needed
OUTPUT_JSON_PATH = "extracted_data.json"

# AWS Clients
# s3_client = boto3.client("s3", region_name=REGION_NAME)
# textract_client = boto3.client("textract", region_name=REGION_NAME)

s3_client = boto3.client("s3", region_name="us-east-1")
textract_client = boto3.client("textract", region_name="us-east-1", endpoint_url="https://textract.us-east-1.amazonaws.com")

def upload_to_s3(local_file: str, bucket: str, s3_key: str):
    """Uploads a file to S3 if it doesn't already exist."""
    try:
        # Check if file already exists in S3
        existing_files = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_key)
        if "Contents" in existing_files:
            print(f"[INFO] File already exists in S3: s3://{bucket}/{s3_key}")
            return

        print(f"[INFO] Uploading {local_file} to S3 bucket {bucket} as {s3_key}...")
        s3_client.upload_file(local_file, bucket, s3_key)
        print("[SUCCESS] Upload completed.")

    except NoCredentialsError:
        print("[ERROR] No AWS credentials found. Configure using `aws configure`.")
        exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to upload file: {e}")
        exit(1)


def process_with_textract(bucket: str, s3_key: str):
    """Uses AWS Textract to extract text from an S3-stored PDF."""
    print("[INFO] Starting Textract document analysis...")

    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}}
        )
        job_id = response["JobId"]
        print(f"[INFO] Textract job started with Job ID: {job_id}")

        # Wait for Textract to finish processing
        while True:
            result = textract_client.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]
            if status in ["SUCCEEDED", "FAILED"]:
                break
            print("[INFO] Waiting for Textract job to complete...")
            time.sleep(5)  # Check every 5 seconds

        if status == "FAILED":
            print("[ERROR] Textract failed to process the document.")
            exit(1)

        # Extract text from Textract response
        extracted_text = "\n".join([block["Text"] for block in result["Blocks"] if block["BlockType"] == "LINE"])
        print("[SUCCESS] Textract completed successfully.")
        return extracted_text

    except ClientError as e:
        print(f"[ERROR] AWS Textract processing failed: {e}")
        exit(1)


def generate_json_with_llm(text: str):
    """Sends the extracted text to an LLM to generate structured JSON."""
    response_schemas = [
        ResponseSchema(name="fund_name", description="The name of the fund"),
        ResponseSchema(name="fees", description="Fees structure in plain text"),
        ResponseSchema(name="performance", description="Any performance figures in the doc"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["pdf_text", "format_instructions"],
        template="""
You are given the following text extracted from a PDF document:
{pdf_text}

Please extract structured data in the following JSON format:
{format_instructions}
"""
    )

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY", "")
    )

    print("[INFO] Sending extracted text to LLM for JSON structuring...")

    # Invoke the prompt using the new LangChain syntax
    response = (prompt | llm).invoke({
        "pdf_text": text,
        "format_instructions": format_instructions
    })

    # Extract text content from AIMessage object
    raw_text = response.content  # Get the actual response content
    print(f"[DEBUG] Raw LLM Response:\n{raw_text}")

    # Strip ```json and ``` formatting if present
    cleaned_text = raw_text.strip("```json").strip("```").strip()

    try:
        json_output = output_parser.parse(cleaned_text)
        print("[SUCCESS] JSON generated successfully.")
        return json_output
    except Exception as e:
        print("[ERROR] Failed to parse JSON:", e)
        print("[DEBUG] Cleaned LLM Response:\n", cleaned_text)
        exit(1)


def generate_json_with_llm_old(text: str):
    """Sends the extracted text to an LLM to generate structured JSON."""
    response_schemas = [
        ResponseSchema(name="fund_name", description="The name of the fund"),
        ResponseSchema(name="fees", description="Fees structure in plain text"),
        ResponseSchema(name="performance", description="Any performance figures in the doc"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["pdf_text", "format_instructions"],
        template="""
You are given the following text extracted from a PDF document:
{pdf_text}

Please extract structured data in the following JSON format:
{format_instructions}
"""
    )

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY", "")
    )
    # chain = LLMChain(llm=llm, prompt=prompt)
    #
    # print("[INFO] Sending extracted text to LLM for JSON structuring...")
    # response = chain.run({
    #     "pdf_text": text,
    #     "format_instructions": format_instructions
    # })
    response = prompt | llm
    print("[INFO] Sending extracted text to LLM for JSON structuring...")
    response = response.invoke({
        "pdf_text": text,
        "format_instructions": format_instructions
    })

    try:
        json_output = output_parser.parse(response)
        print("[SUCCESS] JSON generated successfully.")
        return json_output
    except Exception as e:
        print("[ERROR] Failed to parse JSON:", e)
        print("Raw LLM response:\n", response)
        exit(1)


def save_json_to_file(json_data, output_path="output.json"):
    """Saves JSON output to a local file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"[INFO] JSON saved to {output_path}")


def delete_s3_file(bucket: str, s3_key: str):
    """Deletes the file from S3 after processing (optional)."""
    try:
        s3_client.delete_object(Bucket=bucket, Key=s3_key)
        print(f"[INFO] Deleted {s3_key} from S3.")
    except Exception as e:
        print(f"[WARNING] Failed to delete S3 file: {e}")


def main():
    # 1) Upload the file to S3
    upload_to_s3(LOCAL_PDF_PATH, BUCKET_NAME, S3_FILE_KEY)

    # 2) Process the document with AWS Textract
    extracted_text = process_with_textract(BUCKET_NAME, S3_FILE_KEY)

    # 3) Send extracted text to LLM for JSON conversion
    json_output = generate_json_with_llm(extracted_text)

    # 4) Save JSON output locally
    save_json_to_file(json_output, OUTPUT_JSON_PATH)

    # 5) Optionally delete the file from S3 after processing
    # Uncomment the next line if you want the file removed from S3
    # delete_s3_file(BUCKET_NAME, S3_FILE_KEY)


if __name__ == "__main__":
    main()
