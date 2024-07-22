import fitz  # PyMuPDF
from PIL import Image
import os

# Function to convert PDF to JPEG
def pdf_to_jpg(pdf_path, output_folder):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate over each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load the page
        pix = page.get_pixmap()  # Render the page to an image
        # Save the image in JPEG format
        output_path = os.path.join(output_folder, f'page_{page_num + 1}.jpg')
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(output_path, "JPEG")
        print(f'Saved: {output_path}')
    pdf_document.close()

# Example usage
pdf_path = 'donut_ocr/data/diagnostic_report.pdf'
output_folder = 'donut_ocr/data'
pdf_to_jpg(pdf_path, output_folder)
