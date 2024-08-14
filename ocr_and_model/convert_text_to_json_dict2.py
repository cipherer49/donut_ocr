import re
import json
from collections import defaultdict

def convert_to_json(text):
    patient_info = defaultdict(dict)
    
    # Splitting the text into sections
    sections = re.split(r'\*\*([\w\s]+)\*\*', text)
    
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_content = sections[i + 1].strip()
        
        # Splitting section content by lines
        lines = section_content.split('\n')
        current_key = None
        
        for line in lines:
            # Matching lines with key-value pairs
            match = re.match(r'\*\s*(.+?):\s*(.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if section_title not in patient_info:
                    patient_info[section_title] = {}
                patient_info[section_title][key] = value
            else:
                if line.startswith("*"):
                    current_key = line.replace("*", "").strip()
                    patient_info[section_title][current_key] = {}
                elif line.startswith("+") and current_key:
                    sub_match = re.match(r'\+\s*(.+?):\s*(.+)', line)
                    if sub_match:
                        sub_key = sub_match.group(1).strip()
                        sub_value = sub_match.group(2).strip()
                        patient_info[section_title][current_key][sub_key] = sub_value
    
    # Converting defaultdict to a normal dict
    return {"patient info": dict(patient_info)}

# Function to read the text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Path to the input text file
input_file = "ocr_and_model/haematology2_dddc999b-d54b-4ed9-b175-3d8447a85b88.txt"

# Read text from the file
text = read_text_file(input_file)

# Convert text to JSON structure
json_structure = convert_to_json(text)

# Convert JSON structure to a formatted string
json_str = json.dumps(json_structure, ensure_ascii=False, indent=4)

# Save the JSON string to a text file
output_file = "ocr_and_model/health_fields_output2.json"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_str)

print(f"Converted JSON saved to {output_file}")
