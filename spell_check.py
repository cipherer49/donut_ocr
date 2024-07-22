

import json
from spellchecker import SpellChecker

# Initialize the spell checker
spell = SpellChecker()

# Load the JSON file
json_file_path = 'data/page_6.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Extract the text lines from the JSON data
text_lines = data["text"]

# Process each line
processed_lines = []
for line in text_lines:
    words_in_line = line.split()
    processed_line = []
    for word in words_in_line:
        # Check if the word is in the spell checker's dictionary
        if word.lower() in spell:
            processed_line.append(word)
        else:
            # Correct the spelling of the word if it's misspelled and matches an English word
            corrected_word = spell.correction(word)
            # Ensure corrected_word is not None and it's a valid correction
            if corrected_word and corrected_word.lower() in spell:
                processed_line.append(corrected_word)
            else:
                # Keep the original word if it doesn't match any English word
                processed_line.append(word)
    processed_lines.append(' '.join(processed_line))

# Update the JSON data with the processed text lines
data["text"] = processed_lines

# Save the updated JSON data to the file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)

# Print the processed lines
print("Processed Lines:")
for line in processed_lines:
    print(line)

print(f"Updated text saved to {json_file_path}")
