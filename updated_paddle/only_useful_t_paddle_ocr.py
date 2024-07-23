import json
from paddleocr import PaddleOCR

# Initialize PaddleOCR with language support (e.g., English)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image path for the document
document_image_path = 'data/ht_case5.jpg'

# Perform OCR on the document image
document_result = ocr.ocr(document_image_path, cls=True)

# Function to group words by their y-axis position
def group_by_y_axis(results, threshold=10):
    groups = []
    current_group = []

    for line in results:
        for word_info in line:
            word_text = word_info[1][0]
            word_position = word_info[0]

            if not current_group:
                current_group.append((word_text, word_position))
            else:
                last_word_position = current_group[-1][1]
                if abs(word_position[0][1] - last_word_position[0][1]) < threshold:
                    current_group.append((word_text, word_position))
                else:
                    groups.append(current_group)
                    current_group = [(word_text, word_position)]
    
    if current_group:
        groups.append(current_group)
    
    return groups

# Group words by their y-axis positions
grouped_results = group_by_y_axis(document_result)

# Combine words in each group into a single line, sorted by x-axis position
document_lines = []
for group in grouped_results:
    sorted_group = sorted(group, key=lambda x: x[1][0][0])  # Sort by x-axis position
    line_text = " ".join([word[0] for word in sorted_group])
    document_lines.append(line_text)

# Save the extracted lines into a JSON file
document_json = {"text": document_lines}
json_file_path = 'et_document_text_ht_case5.json'
with open(json_file_path, 'w') as json_file:
    json.dump(document_json, json_file, indent=4)

print(f"Extracted text saved to {json_file_path}")
