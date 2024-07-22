import json



from paddleocr import PaddleOCR
import json

# Initialize PaddleOCR with language support (e.g., English)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image path for the document
document_image_path = 'data/page_6.jpg'

# Perform OCR on the document image
document_result = ocr.ocr(document_image_path, cls=True)

# Extract the text from the OCR result
document_texts = []
for line in document_result:
    for word_info in line:
        document_texts.append(word_info[1][0])

# Save the extracted text into a JSON file
document_json = {"text": document_texts}
json_file_path = 'document_text.json'
with open(json_file_path, 'w') as json_file:
    json.dump(document_json, json_file, indent=4)

print(f"Extracted text saved to {json_file_path}")


