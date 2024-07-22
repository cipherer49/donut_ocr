import json
import os
from paddleocr import PaddleOCR, draw_ocr


#defining image path and json path(to store extract data)

#image path(jpg,png)1
img = 'data/reimbursement/page_1.jpg'

#saving printed text json_file path1
et_json_path = 'data/reimbursement/paddle_page_1.json'


class ocr_image_and_pdf ():
   
        


    #defining function for that
    def printed_text(self):
        # Initialize PaddleOCR with language support (e.g., English)
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # Perform OCR on the document image
        document_result = ocr.ocr(img, cls=True)

        # Extract the text from the OCR result
        document_texts = []
        for line in document_result:
            for word_info in line:
                document_texts.append(word_info[1][0])

        # Save the extracted text into a JSON file
        document_json = {"text": document_texts}
        
        with open(et_json_path, 'w') as json_file:
            json.dump(document_json, json_file, indent=4)

        print(f"Extracted text saved to {et_json_path}")

    def hand_written_text(self):
        #performing ocr on handwritten image
        ocr =  PaddleOCR(use_angle_cls=True, lang='en')

        #performing ocr on handwritten image
        hand_written_result = ocr.ocr(img, cls=True)

        # Extract the text from the OCR result
        hand_written_texts = []
        for line in hand_written_result:
            for word_info in line:
                hand_written_texts.append(word_info[1][0])

        # Save the extracted text into a JSON file
        document_json = {"text": hand_written_texts}
        
        with open(et_json_path, 'w') as json_file:
            json.dump(document_json, json_file, indent=4)

        print(f"Extracted text saved to {et_json_path}")

    #to extract data from json file
    def get_data(self):
    
        with open(et_json_path, 'r') as json_file:
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
        got_data_path = "data/reimbursement/paddle_query_page_1.json"
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
                self.get_data()
            elif ask_again == 'n':
                return "ok"
            else:
                return "wrong input"
        elif show_opt == '2':
            print(f'The query data will be saved in this path: {got_data_path}')
            with open(got_data_path, 'w') as json_file:
                json.dump(updated_data, json_file, indent=4)
            print(f"The query data was saved to {got_data_path}")

            ask_again = input("Do you want to ask another query (y/n): ")
            if ask_again == 'y':
                self.get_data()
            elif ask_again == 'n':
                return "ok"
            else:
                return "wrong input"
    
    def option(self):
        #for choosing what type of ocr
        self.ocr_opt = input("for ht_ocr  input:1 \n  for pt_ocr input:2 \n you're input here:")
        
        #defining option
        
        
        if self.ocr_opt == '1':
            print("running ht_ocr")
            
            self.hand_written_text()

            ask_get_data = input("do you want to get or retrieve some fields(y/n):")
            if ask_get_data == 'y':
                
                self.get_data()
                return "**the end**"
            elif ask_get_data == 'n':
                print("ok code complete")
            else:
                print("check input !")
        elif self.ocr_opt == '2':
            print("running pt_ocr")
            self.printed_text()

            ask_get_data = input("do you want to get or retrieve some fields(y/n):")
            if ask_get_data == 'y':
                self.get_data()
                return "**the end**"
            elif ask_get_data == 'n':
                print("ok code complete")
            else:
                print("check input !")
        else:
            print("incorrect input")



        

        

#running the class
run = ocr_image_and_pdf()
run.option()




        

