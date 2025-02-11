import subprocess
import pytesseract
from pdf2image import convert_from_path

def main():
    print("Testing Tesseract availability...")
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        print("Tesseract info:")
        print(result.stdout)
    except FileNotFoundError:
        print("Error: 'tesseract' not found on PATH.")

    print("\nTesting pytesseract path:")
    print("pytesseract.pytesseract.tesseract_cmd =", pytesseract.pytesseract.tesseract_cmd)

    # Optionally set a custom path if needed:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Path\to\tesseract.exe"

    print("\nTesting pdf2image + poppler:")
    pdf_path = "data/input/mw/28.03.2024 Factsheet Lumyna - MW TOPS UCITS Fund GBP B (acc).pdf"

    try:
        # 1) Convert PDF pages to images using pdf2image
        images = convert_from_path(pdf_path, dpi=72)
        print(f"Successfully converted PDF to {len(images)} images.")

        # 2) Perform OCR on each image
        for idx, page_img in enumerate(images, start=1):
            ocr_text = pytesseract.image_to_string(page_img)
            print(f"\n--- OCR text from page {idx} ---\n{ocr_text}\n")

    except Exception as e:
        print(f"Error converting or OCR-ing PDF: {e}")


if __name__ == "__main__":
    main()
