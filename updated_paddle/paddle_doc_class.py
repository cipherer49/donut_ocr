from paddleocr import PaddleOCR
import cv2

# Initialize PaddleOCR with the desired language model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Path to the document image
image_path = 'data/more_files/apollo_pharma_1.jpg'

# Perform OCR on the image
ocr_result = ocr.ocr(image_path, cls=True)

# Extract recognized text
recognized_text = " ".join([line[1][0] for line in ocr_result[0]])


# Document Classification Function
def classify_document(text):
    if "invoice" in text.lower():
        return "Invoice"
    elif "resume" in text.lower():
        return "Resume"
    elif "report" in text.lower():
        return "Report"
    else:
        return "Unknown Document Type"

# Classify the document based on recognized text
document_type = classify_document(recognized_text)
print(f"Document Type: {document_type}")



