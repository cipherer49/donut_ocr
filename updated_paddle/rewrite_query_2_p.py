import os
import json
import difflib

# Load the JSON file
json_file_path = 'et_document_text_ht_case5.json'

def get_data():
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract the text lines from the JSON data
    text_lines = data["text"]

    # Prompt word to search for (case-insensitive)
    prompt_query = input("What do you want to extract: ").lower()

    # Split the query into individual words
    query_words = prompt_query.split()

    # Function to check if a line contains all words or close matches to the query words
    def contains_query_words(line, query_words):
        words = line.lower().split()
        for query_word in query_words:
            if not any(difflib.SequenceMatcher(None, word, query_word).ratio() > 0.69 for word in words):
                return False
        return True

    # Find lines that contain all words or close matches to the query words
    extracted_lines = [line for line in text_lines if contains_query_words(line, query_words)]

    # Load existing extracted text if the file exists
    got_data_path = "paddle_query_ht_case5.json"
    if os.path.exists(got_data_path):
        with open(got_data_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    # Create a unique key for the new query
    query_key = f"full query {prompt_query}"

    # Append new extracted lines to the existing data under the new query key
    existing_data[query_key] = extracted_lines

    # Prepare the updated JSON data
    updated_data = existing_data

    # Ask if the user wants to print in terminal or save in JSON
    show_opt = input("To print in terminal input: 1 \nTo save in JSON file input: 2 \nYour input here: ")

    if show_opt == '1':
        print("Extracted Lines:")
        for line in extracted_lines:
            print(line)

        ask_again = input("Do you want to ask another query (y/n): ")
        if ask_again == 'y':
            get_data()
        elif ask_again == 'n':
            print("Ok")
        else:
            print("Wrong input")
    elif show_opt == '2':
        print(f'The query data will be saved in this path: {got_data_path}')
        with open(got_data_path, 'w') as json_file:
            json.dump(updated_data, json_file, indent=4)
        print(f"The query data was saved to {got_data_path}")

        ask_again = input("Do you want to ask another query (y/n): ")
        if ask_again == 'y':
            get_data()
        elif ask_again == 'n':
            print("Ok")
        else:
            print("Wrong input")

# Running the function
get_data()
