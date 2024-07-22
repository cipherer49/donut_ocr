import os 
import json

# Load the JSON file
json_file_path = 'data/paddle_page_15.json'
def get_data():
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract the text lines from the JSON data
    text_lines = data["text"]

    # Prompt word to search for (case-insensitive)
    prompt_word = input("What do you want to extract: ").lower()

    # Find lines that contain the prompt word (case-insensitive) and the following line if it exists
    extracted_lines = []
    for i, line in enumerate(text_lines):
        if prompt_word in line.lower():
            extracted_lines.append(line)  # Always append the matched line
            # Check if the next line exists before appending
            if i + 1 < len(text_lines):
                extracted_lines.append(text_lines[i + 1])

    # Load existing extracted text if the file exists
    got_data_path = "data/paddle_query_page_15.json"
    if os.path.exists(got_data_path):
        with open(got_data_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    # Create a unique key for the new query
    query_key = f"full query {prompt_word}"

    # Append new extracted lines to the existing data under the new query key
    existing_data[query_key] = extracted_lines

    # Prepare the updated JSON data
    updated_data = existing_data

    # Ask if the user wants to print in terminal or save in JSON
    show_opt = input("To print in terminal input :1 \nTo save in JSON file input: 2 \nYour input here: ")

    if show_opt == '1':
        print("Extracted Lines:")
        for line in extracted_lines:
            print(line)

        ask_again = input("Do you want to ask another query (y/n): ")
        if ask_again == 'y':
            get_data()
        elif ask_again == 'n':
            print("ok")
        else:
            print( "wrong input")
    elif show_opt == '2':
        print(f'The query data will be saved in this path: {got_data_path}')
        with open(got_data_path, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4)
        print(f"The query data was saved to {got_data_path}")

        ask_again = input("Do you want to ask another query (y/n): ")
        if ask_again == 'y':
            get_data()
        elif ask_again == 'n':
            print("ok")
        else:
            print("wrong input")

#running
get_data()