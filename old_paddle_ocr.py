from paddleocr import PaddleOCR, draw_ocr

# Initialize PaddleOCR with language support (e.g., English)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image paths for document and handwritten text
document_image_path = 'data/page_6.jpg'
#handwritten_image_path = 'data/hand_written2.jpg'

# Perform OCR on document image
document_result = ocr.ocr(document_image_path, cls=True)

# Perform OCR on handwritten image
#handwritten_result = ocr.ocr(handwritten_image_path, cls=True)

# Print recognized text
print("Document Text:")
for line in document_result:
    print(line)

# print("\nHandwritten Text:")
# for line in handwritten_result:
    # print(line)