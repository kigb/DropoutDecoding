import argparse
import shutil
from datetime import datetime
import json
import os
from random import sample, seed
import subprocess
import numpy as np
from tqdm import tqdm
# from transformers import LlavaNextProcessor
from models.utils import CustomLlavaNextForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from transformers import LlavaForConditionalGeneration, \
    AutoProcessor
from models.llava import CustomLlavaForConditionalGeneration
import requests
from PIL import Image
from io import BytesIO

def load_coco_data(data_dir):
    annotation_file_path = data_dir + "annotations/instances_val2014.json"
    caption_file_path = data_dir + "annotations/captions_val2014.json"
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    coco = COCO(caption_file_path)
    return coco, coco_anns

def prepare_pope_data():
    """
    Prepare the data for the POPE generation.
    """
    # Enter the directory and execute the script
    script_directory = "/home/fyx/hallucination/pope_test/pope_metric"
    script_name = "main.py"
    
    # Construct the full path to the script
    script_path = os.path.join(script_directory, script_name)
    
    # Run the script using the current Python interpreter
    try:
        result = subprocess.run(
            ["python", script_path], 
            cwd=script_directory,  # Change to the target directory
            check=True  # Raises an error if the command fails
        )
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during script execution: {e}")

def parse_pope_file(file_path):
    """
    parse pope file and store image and text(prompt) information
    """
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            image_text_pair = {
                "image": data["image"],
                "text": data["text"]
            }
            result.append(image_text_pair)
    return result

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def save_ans(ans, question, answer):
    image_ans_pair = {
        "question": question,
        "answer": answer
    }
    ans.append(image_ans_pair)

def evaluate(ans_file,label_file, number = 3000):
    #load first 300
    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    answers = answers[:number]
    label_list = label_list[:number]
    for answer in answers:
        text = answer['answer']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def main(args):
    # ----------------- Load Model -----------------
    if args.model == "llava-next":
        model_path = "/data3/fyx/llava-v1.6-mistral-7b-hf"
    elif args.model == "llava":
        model_path = "/data3/fyx/llava-1.5-7b-hf"
    if args.model == "llava-next":
        processor = LlavaNextProcessor.from_pretrained(model_path)
    elif args.model == "llava":
        processor = AutoProcessor.from_pretrained(model_path)
    device = 'cuda:2'
    if args.original is True:
        print("generating original")
        if args.model == "llava-next":
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map = device
            )
        elif args.model == "llava":
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device
            )
    else:
        if args.model == "llava-next":
            model = CustomLlavaNextForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map = device
            )
        elif args.model == "llava":
            model = CustomLlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device
            )

    model = model.to(device)
    
    # ----------------- Load Data -----------------
    image_base_path = "/data3/fyx/COCO/val2014/"
    if args.refresh_data is True:
        prepare_pope_data()
        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        src_dir = "/home/fyx/hallucination/pope_test/pope_metric/output/coco/"
        dst_dir = f"/home/fyx/hallucination/pope_test/pope_samples/{cur_time}/coco/"

        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        # Copy files from source to destination
        for file_name in os.listdir(src_dir):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dst_dir)

        print(f"Files copied to {dst_dir}")
    # popefilepath can also be random or popular
    pope_file_pathes = ["/home/fyx/hallucination/pope_test/pope_metric/output/coco/coco_pope_adversarial.json", "/home/fyx/hallucination/pope_test/pope_metric/output/coco/coco_pope_popular.json","/home/fyx/hallucination/pope_test/pope_metric/output/coco/coco_pope_random.json" ]
    for pope_file_path in pope_file_pathes:
        print(f"the pope file is {pope_file_path}")
        pope_data = parse_pope_file(pope_file_path)
        print(len(pope_data))
        ans = []
        # for i in tqdm(range(len(pope_data))):
        for i in tqdm(range(args.number)):
            image_path = image_base_path + pope_data[i]["image"]
            text = pope_data[i]["text"]
            image = load_image(image_path)
            if args.model == "llava-next":
                prompt = f"[INST] <image>\n{text}[/INST]"
            elif args.model == "llava":
                prompt = f"USER: <image>\n{text} ASSISTANT:"
            inputs = processor(prompt, image, return_tensors="pt").to(device)
            if args.original is True:
                output_ids = model.generate(**inputs, max_new_tokens=1, num_beams=1,
                                            pad_token_id=processor.tokenizer.eos_token_id)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=1, use_input_embeddings=False,num_beams=1,pad_token_id=processor.tokenizer.eos_token_id)
            # output_ids = model.generate(**inputs, max_new_tokens=1, use_input_embeddings=False,num_beams=1,pad_token_id=processor.tokenizer.eos_token_id)
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
            if args.model == "llava-next":
                output_text = output_text[0].split('[/INST]', 1)[-1].strip()
            elif args.model == "llava":
                output_text = output_text[0].split('ASSISTANT:', 1)[-1].strip()
            # print(output_text)
            save_ans(ans, text, output_text)
        ans_path = "/home/fyx/hallucination/pope_test/pope_metric/answer/"
        # file name is timestamp_ans.json
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        ans_file_name = timestamp + "_ans.json"
        with open(ans_path + ans_file_name, 'w') as file:
            for item in ans:
                json_str = json.dumps(item)
                file.write(json_str + "\n")
        print("Answer saved successfully.")
        print(f"Answer file: {ans_file_name}")
        # ----------------- Evaluate -----------------

        evaluate(ans_path + ans_file_name, pope_file_path,args.number)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-next")
    parser.add_argument("--original",type=bool, default=False)
    parser.add_argument("--refresh-data", type=bool, default=False)
    parser.add_argument("--number", type=int, default=3000)
    args = parser.parse_args()
    main(args)