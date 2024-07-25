from paddleocr import PaddleOCR
import json

# Initialize PaddleOCR with table structure recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec=False, det=True, structure=True)

def process_image(image_path):
    """
    Process the image to perform OCR and table recognition.
    """
    result = ocr.ocr(image_path, cls=True)
    return result

def extract_key_value_pairs(result):
    """
    Extract key-value pairs from the OCR result.
    """
    key_value_pairs = {}

    for line in result:
        for word_info in line:
            text = word_info[1][0]  # Extract the text from the OCR result
            
            # Simple example logic for identifying key-value pairs
            if ':' in text:
                key, value = text.split(':', 1)
                key_value_pairs[key.strip()] = value.strip()

    return key_value_pairs

def save_to_json(data, file_path):
    """
    Save the extracted data to a JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main(image_path, json_file_path):
    """
    Main function to process the image, extract data, and save to JSON file.
    """
    result = process_image(image_path)
    key_value_pairs = extract_key_value_pairs(result)
    save_to_json(key_value_pairs, json_file_path)
    print(f"Data saved to {json_file_path}")

# Example usage
if __name__ == "__main__":
    image_path = 'data/page_15.jpg'  # Replace with your image path
    json_file_path = 'output_data.json'  # Replace with your desired output file path
    main(image_path, json_file_path)
