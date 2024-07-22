import torch
from PIL import Image
import re
import json
from transformers import DonutProcessor,VisionEncoderDecoderModel
image = Image.open("data/page_6.jpg").convert('RGB')

json_path = "data/page_6.json"


class donut_ocr():
    def __init__(self):
        self.outputs = {}

    def  classify_doc(self,image):
        # load the processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        #load the model
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        #print(torch.cuda.is_available())

        #preparing the image
        pixel_values = processor(image,return_tensors="pt").pixel_values

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
        #inserting the class output in  json
        self.outputs['class'] = self.result
        #just printing the class result
        print(self.result['class'])

    def parsing_doc(self,image):

        #writing a if  statement to check  if file class matches
        if self.result == {"class": "invoice"} or self.result == {"class":"form"} or self.result == {"class":"scientific_report"} or self.result=={"class":"specification"}:

            # load the processor
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            #load the model
            model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            #print(torch.cuda.is_available())

            #preparing the image
            pixel_values = processor(image,return_tensors="pt").pixel_values

            #create prompt for document parsing task
            task_prompt = "<s_cord-v2>"
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
            parse_result = processor.token2json(seq)
            #writing in json file
            self.outputs["parse_output"] = parse_result
            print(parse_result)
        else:
            print("the document class doesn't match")
    
    def writing_in_json(self,file_name):
         with open(file_name, 'w') as file:
            json.dump(self.outputs, file)




#running the instance

run = donut_ocr()
run.classify_doc(image)
run.parsing_doc(image)
run.writing_in_json(json_path)