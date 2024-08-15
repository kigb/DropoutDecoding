import argparse
from datetime import datetime
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import LlavaNextProcessor
from models.utils import CustomLlavaNextForConditionalGeneration
import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_coco_data(data_dir):
    annotation_file_path = data_dir + "annotations/instances_val2014.json"
    caption_file_path = data_dir + "annotations/captions_val2014.json"
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    coco = COCO(caption_file_path)
    return coco, coco_anns

def main(args):
    # load model
    model_path = "/data3/fyx/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path)
    device = 'cuda:3'
    model = CustomLlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map = device
    )

    # COCO dataset
    coco, coco_anns = load_coco_data("/data3/fyx/COCO/")
    img_ids = coco.getImgIds()
    # ---------begin prepare sample dataset---------
    num_samples = 200 # number of samples
    step = 100 # step size, sample every step images
    num_samples = min(num_samples, len(img_ids) // step)
    sampled_indices = [i for i in range(0, len(img_ids), step)]
    sampled_indices = sampled_indices[:num_samples]
    sampled_img_ids = [img_ids[i] for i in sampled_indices]
    img_files = []
    for cur_img_id in sampled_img_ids:
        cur_img = coco.loadImgs(cur_img_id)[0]
        print(cur_img)
        cur_img_path = cur_img["file_name"]
        img_files.append(cur_img_path)
    img_dict = {}
    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}
    img_dict = {}
    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}
    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}
    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )
    # ---------end prepare sample dataset---------

    # ---------begin prepare output data dir---------
    base_dir = os.path.join("/home/fyx/vlm_outputs")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    #ENDCOCOTEST_LOADINGINDEX
    now = datetime.now()
    t = now.strftime("%m%d%H%M")
    filename = args.save_name + t + ".json"
    # ---------end prepare output data dir---------
    for idx, img_id in tqdm(enumerate(range(len(img_files))), total=len(img_files)):
        
        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])

        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id
        
        with open("/home/fyx/llava_masked_tokens.txt", "a") as f:
            # write the img_file to the file
            f.write(img_file)
            f.write("\n")
        f.close()

        # begin process input data
        image_path = "/data3/fyx/COCO/val2014/" + img_file
        image = load_image(image_path)
        prompt = "[INST] <image>\nDescribe the image in detail [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=100, use_input_embeddings=False)
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
        output_text = output_text[0].split('[/INST]', 1)[-1].strip()
        sentence_list = output_text.split(".")
        sentence_filter_list = []
        for sentence in sentence_list:
            if "unk" not in sentence:
                sentence_filter_list.append(sentence)
        output_text = ".".join(sentence_filter_list)
        print("decoder output text", output_text)
        img_save["caption"] = output_text
        print("image_path: ", image_path)
        print("caption: ", output_text)
        #获取时间
        
        generated_captions_path = os.path.join(
            base_dir,
            filename)
        # print("generated_captions_path", generated_captions_path)
        with open(generated_captions_path, "a") as f:
            json.dump(img_save, f)
            f.write("\n")
    
    # -------- begin json data eval --------
    loaded_json = []

    generated_captions_path = "/home/fyx/vlm_outputs/"
    generated_captions_path = generated_captions_path + filename
    with open(generated_captions_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            loaded_json.append(json.loads(line))

    # eliminate the items in loaded_json with the same key:
    for i in range(len(loaded_json)):
        for j in range(i + 1, len(loaded_json)):
            if loaded_json[i]["image_id"] == loaded_json[j]["image_id"]:
                loaded_json.pop(j)
                break

    print("loaded_json:", len(loaded_json))

    # construct output file as input to CHAIR evaluation
    # output format follows https://github.com/ruotianluo/self-critical.pytorch
    formulated_output_dict = {}
    # overall result
    all_overall_scores = defaultdict(list)
    # imgToEval per image result
    img_to_eval_dict = {}
    # to save memory, load 100 captions at a time
    for start_idx in tqdm(range(0, len(loaded_json), 100), desc="Generating CHAIR Input"):
        # define the current iteration end index
        end_idx = min(start_idx + 100, len(loaded_json))
        coco_res = coco.loadRes(
            loaded_json[start_idx:end_idx],
        )
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.params["image_id"] = coco_res.getImgIds()
        coco_eval.evaluate()

        # keep track of the overall scores
        for metric, score in coco_eval.eval.items():
            all_overall_scores[metric].append(score)

        # imgToEval per image result
        for i, cur_img_id in enumerate(coco_res.getImgIds()):
            cur_eval_dict = coco_eval.evalImgs[i]
            # add caption to the eval dict
            cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
            img_to_eval_dict[cur_img_id] = cur_eval_dict

    # overall result
    overall_dict = {}
    for metric, score in all_overall_scores.items():
        overall_dict[metric] = np.mean(score)
    formulated_output_dict["overall"] = overall_dict
    formulated_output_dict["imgToEval"] = img_to_eval_dict


    print(f"\nGenerated {len(img_to_eval_dict)} samples results in CHAIR format.")

    # save the formulated output dict
    formulated_output_path = "/home/fyx/vlm_results/"
    formulated_output_path = formulated_output_path + args.save_name 

    with open(formulated_output_path, "w") as f:
        json.dump(formulated_output_dict, f)
    
    print("output file saved at: ", formulated_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-name",type=str, default="output")
    args = parser.parse_args()
    main(args)