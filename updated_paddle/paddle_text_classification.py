from paddleocr import PaddleOCR, PPStructure, draw_structure_result, save_structure_res
import matplotlib.pyplot as plt
from PIL import Image
import os

# Initialize the OCR system with document structure analysis
ocr = PaddleOCR(use_angle_cls=True, lang='en')
structure = PPStructure(recovery=True)

# Path to the document image
image_path = 'data/more_files/apollo_pharma_invoice/apollo_pharma_1.jpg'

# Use PaddleOCR's default font
default_font_path = os.path.join(os.path.dirname(__file__), 'paddleocr', 'ppocr', 'utils', 'simsun.ttc')

# Perform document structure analysis
result = structure(image_path)

# Extract and display the classification result
for item in result:
    if 'classification' in item:
        print(f"Document Type: {item['classification']}")

# Optionally, visualize the structure analysis
image = Image.open(image_path).convert('RGB')
draw_img = draw_structure_result(image, result, font_path=default_font_path)
im_show = Image.fromarray(draw_img)

# Save or display the result
im_show.save('structured_result.jpg')
plt.imshow(im_show)
plt.show()
