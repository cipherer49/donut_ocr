import json
import os
import difflib
from paddleocr import PaddleOCR

#path of images and json
img_path = 'data/page_3.jpg'
ocr_json_path = 'data/new_paddle_ext_data/check_new_page_3.json'
ocr_query_json_path = 'data/new_paddle_ext_data/query_new_page_3.json'
#writing a class to fit all functions
class all_func():




    #writing function to do any ocr(handwritten and printed text)

    def all_ocr(self):
        # Initialize PaddleOCR with language support (e.g., English)
        ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=True)

        #perform ocr on the image 
        ocr_result = ocr.ocr(img_path,cls =True)
            #writing function to group text by their y-axis position
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
        grouped_results = group_by_y_axis(ocr_result)


        # Combine words in each group into a single line, sorted by x-axis position
        ocr_lines = []
        for group in grouped_results:
            sorted_group = sorted(group, key=lambda x: x[1][0][0])  # Sort by x-axis position
            line_text = " ".join([word[0] for word in sorted_group])
            ocr_lines.append(line_text)

        # Save the extracted lines into a JSON file
        ocr_json = {"text": ocr_lines}
        
        with open(ocr_json_path, 'w') as json_file:
            json.dump(ocr_json, json_file, indent=4)

        print(f"Extracted text saved to {ocr_json_path}")
    
    #writing function to get query from ocr_json_path

    def get_data(self):
        prompt_ask = input("Do you want to retrieve some fields(y/n):")
        if prompt_ask == 'y':

            with open(ocr_json_path, 'r') as json_file:
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
            
            if os.path.exists(ocr_query_json_path):
                with open(ocr_query_json_path, 'r') as json_file:
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
                    self.get_data()
                elif ask_again == 'n':
                    print("Ok")
                else:
                   print("Wrong input, try again")
                   self.get_data()
            elif show_opt == '2':
                print(f'The query data will be saved in this path: {ocr_query_json_path}')
                with open(ocr_query_json_path, 'w') as json_file:
                    json.dump(updated_data, json_file, indent=4)
                print(f"The query data was saved to {ocr_query_json_path}")

                ask_again = input("Do you want to ask another query (y/n): ")
                if ask_again == 'y':
                    self.get_data()
                elif ask_again == 'n':
                    print("Ok")
                else:
                    print("Wrong input, try again")
                    self.get_data()
        elif prompt_ask == 'n':
            print("ok code completed")
        else:
            print("incorrect option check input and try again")
            self.get_data()


#running the code
run = all_func()

run.all_ocr()
run.get_data()








    


