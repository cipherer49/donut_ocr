import json
import torch
import os
import difflib
import re
from collections import Counter
from paddleocr import PaddleOCR
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Default paths for images and JSON
default_img_path = "ocr_and_model/test_data/"
ocr_json_path = 'ocr_and_model/test_data/haematology_paddle+class.json'

class AllFunc:
    def __init__(self, img_path):
        self.img_path = img_path
        self.ocr_lines = []
        self.class_counter = Counter()

    def classifier_donut(self, img):
        # Load the processor
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        # Load the model
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

        # Preparing the image
        pixel_values = processor(img, return_tensors="pt").pixel_values

        # Create prompt for document classification task
        task_prompt = "<s_rvlcdip>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Generate output
        outputs = model.generate(pixel_values.to(device),
                                 decoder_input_ids=decoder_input_ids.to(device),
                                 max_length=model.decoder.config.max_position_embeddings,
                                 early_stopping=True,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 eos_token_id=processor.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True,
                                 output_scores=True)

        # Clean the response
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()

        # Convert the response to JSON
        result = processor.token2json(seq)
        class_name = next(iter(result.values()))
        self.class_counter[class_name] += 1

    def all_ocr(self, img_path):
        # Initialize PaddleOCR with language support (e.g., English)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

        # Perform OCR on the image
        ocr_result = ocr.ocr(img_path, cls=True)

        # Function to group text by their y-axis position
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
        for group in grouped_results:
            sorted_group = sorted(group, key=lambda x: x[1][0][0])  # Sort by x-axis position
            line_text = " ".join([word[0] for word in sorted_group])
            self.ocr_lines.append(line_text)

    def dump_in_json(self, file_path):
        # Get the most frequent class
        most_frequent_class = self.class_counter.most_common(1)[0][0]

        # Prepare the final JSON structure
        ocr_json = {"class": [most_frequent_class], "text": self.ocr_lines}
        self.outputs = ocr_json

        with open(file_path, 'w') as file:
            json.dump(self.outputs, file, indent=4)
            print(f"Saved in JSON: {file_path}")

    def process_images(self):
        if os.path.isdir(self.img_path):
            # If the path is a directory, process all image files in the directory
            for filename in os.listdir(self.img_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(self.img_path, filename)
                    img = Image.open(full_path).convert('RGB')
                    self.all_ocr(full_path)
                    self.classifier_donut(img)
            # Dump all results into a single JSON file
            self.dump_in_json(ocr_json_path)
        elif os.path.isfile(self.img_path):
            # If the path is a single image file, process it normally
            img = Image.open(self.img_path).convert('RGB')
            self.all_ocr(self.img_path)
            self.classifier_donut(img)
            self.dump_in_json(ocr_json_path)
        else:
            print(f"Invalid path: {self.img_path}")

# Running the code
run = AllFunc(default_img_path)
run.process_images()
