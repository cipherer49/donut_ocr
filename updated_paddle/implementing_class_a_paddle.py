import json
import torch
import os
import difflib
import re
from paddleocr import PaddleOCR
from PIL import Image
from transformers import DonutProcessor,VisionEncoderDecoderModel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#path of images and json
img =  Image.open("data/more_files/doctor_bill/doctor_bill.jpg").convert('RGB')# for  donut  class converted mode = rgb
img_path = "data/more_files/doctor_bill/doctor_bill.jpg" #mode = default path for paddle
ocr_json_path = 'data/more_files/doctor_bill/doctor_bill_paddle+class.json'
ocr_query_json_path = 'data/more_files/doctor_bill/doctor_bill_query.json'
#writing a class to fit all functions
class all_func():
    


    def classifier_donut(self):
         # load the processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        #load the model
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        #print(torch.cuda.is_available())

        #preparing the image
        pixel_values = processor(img,return_tensors="pt").pixel_values

        #create prompt for document classification task
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = processor.tokenizer(task_prompt,add_special_tokens=False,return_tensors="pt")["input_ids"]


        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)

        #generate output
        outputs = model.generate(pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams = 1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True)

        #clean the response
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token,"").replace(processor.tokenizer.pad_token,"")
        seq = re.sub(r"<.*?>","",seq,count=1).strip()

        #convert the reponse to json
        self.result = processor.token2json(seq)
        self.class_lines = []
        self.class_lines.append(self.result)

    #implementing paddle document classifier

    def paddle_doc_classifier(self):
        ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=True)
        # Perform OCR on the image
        ocr_result = ocr.ocr(img_path, cls=True)

        # Extract recognized text
        recognized_text = " ".join([line[1][0] for line in ocr_result[0]])


        # Document Classification Function
        def classify_document(text):
            if "invoice" in text.lower():
                return "Invoice"
            elif "resume" in text.lower():
                return "Resume"
            elif "report" in text.lower():
                return "Report"
            else:
                return "Unknown Document Type"

        # Classify the document based on recognized text
        document_type = classify_document(recognized_text)
        print(f"Document Type: {document_type}")

        self.class_lines = []
        self.class_lines.append(document_type)

        
        
        
        
        






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
        self.ocr_lines = []
        for group in grouped_results:
            sorted_group = sorted(group, key=lambda x: x[1][0][0])  # Sort by x-axis position
            line_text = " ".join([word[0] for word in sorted_group])
            self.ocr_lines.append(line_text)

        # Save the extracted lines into a JSON file
        
        
        

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
    
    def dump_in_json(self,file_path):
        ocr_json = {"class":self.class_lines,"text": self.ocr_lines}
        self.outputs = ocr_json
        with open(file_path, 'w') as file:
            
            json.dump(self.outputs, file,indent=4)
            print("saved in json")
    
    def opt_doc_classifier(self):
        class_ask = input("what document classifier you want(paddle:1,donut:2):")
        if class_ask == '1':
            print("running paddle classifier")

            self.paddle_doc_classifier()
        elif class_ask == '2':
            print("running  donut classifier")
            self.classifier_donut()
        else:
            print("incorrect input")





#running the code
run = all_func()
run.opt_doc_classifier()

run.all_ocr()
run.dump_in_json(ocr_json_path)
run.get_data()



