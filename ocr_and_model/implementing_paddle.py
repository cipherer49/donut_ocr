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
img =  Image.open("data/reimbursement/page_1.jpg").convert('RGB')# for  donut  class converted mode = rgb
img_path = "data/reimbursement/page_1.jpg" #mode = default path for paddle
ocr_json_path = 'data/reimbursement/page_1_paddle+class.json'
ocr_query_json_path = 'data/page_1_query.json'
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
    
    
    def dump_in_json(self,file_path):
        ocr_json = {"class":self.class_lines,"text": self.ocr_lines}
        self.outputs = ocr_json
        with open(file_path, 'w') as file:
            
            json.dump(self.outputs, file,indent=4)
            print("saved in json")
    






#running the code
run = all_func()

run.all_ocr()
run.classifier_donut()
run.dump_in_json(ocr_json_path)




