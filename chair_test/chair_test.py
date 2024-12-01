import argparse
from datetime import datetime
import json
import os
from random import sample, seed
import numpy as np
from tqdm import tqdm
from transformers import LlavaNextProcessor
from models.llava import CustomLlavaForConditionalGeneration
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from models.llavanext import CustomLlavaNextForConditionalGeneration
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
)
from models.instructblip import CustomInstructBlipForConditionalGeneration
import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import requests
from PIL import Image
from io import BytesIO
from collections import defaultdict
from chair_test.chair_metrics import chair
from models.config import settings

prompt_dict = {}
prompt_dict["llava-1.5"] = "USER: <image>\nDescribe the image. ASSISTANT:"
prompt_dict["instructblip"] = "Describe the image."
prompt_dict["llava-next"] = "[INST] <image>\nDescribe the image. [/INST]"


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_coco_data(data_dir):
    annotation_file_path = os.path.join(data_dir, "annotations/instances_val2014.json")
    caption_file_path = os.path.join(data_dir, "annotations/captions_val2014.json")
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    coco = COCO(caption_file_path)
    return coco, coco_anns


def chair_eval(
    chair_input_path,
    model_type,
    num_images,
    output_dir,
    dataset_name,
    data_dir,
    metric,
    verbosity=False,
):
    if verbosity:
        print("\nchair_input_path: ", chair_input_path)

    # sanity check between caption file and command line arguments
    model_name = "llava"
    output_dir = os.path.join(
        output_dir, metric, f"{model_name}_{model_type}", dataset_name
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # annotation path should be under data dir
    annotation_dir = f"{data_dir}/annotations"
    # load the generated captions
    _, imids, _ = chair.load_generated_captions(chair_input_path)
    print("load generation")
    # initialize CHAIR with generated captions and annotations
    evaluator = chair.CHAIR(imids, annotation_dir)
    evaluator.get_annotations()
    # compute chair metrics
    cap_dict = evaluator.compute_chair(chair_input_path)
    # save to json pretty print
    chair_json_path = os.path.join(
        output_dir,
        f"{model_name}_{model_type}_{dataset_name}_num_images_{num_images}_chair_results.json",
    )
    with open(chair_json_path, "w") as f:
        json.dump(cap_dict, f, indent=4)
    # print metric
    metric_string_ce = chair.print_metrics(cap_dict, quiet=False)

    # save results
    result_path = os.path.join(
        output_dir,
        f"{model_name}_{model_type}_{dataset_name}_num_images_{num_images}_chair_results.txt",
    )
    with open(result_path, "w") as f:
        f.write(metric_string_ce)
    if verbosity:
        print(f"\nCHAIR results saved to {result_path}.")

    halc_caption_result = cap_dict["sentences"]
    halc_result = {}
    for i in halc_caption_result:
        halc_result[i["image_id"]] = {
            "caption": i["caption"],
            "cider": max(np.log10(i["metrics"]["CIDEr"]) + 20, 0),
            "meteor": i["metrics"]["METEOR"],
            "chairs": i["metrics"]["CHAIRs"],
            "chairi": i["metrics"]["CHAIRi"],
            "bleu": (
                i["metrics"]["Bleu_1"]
                + i["metrics"]["Bleu_2"]
                + i["metrics"]["Bleu_3"]
                + i["metrics"]["Bleu_4"]
            )
            / 4,
            "objects_num": len(i["mscoco_generated_words"]),
            "words_num": len(i["words"]),
            "hallucinate_num": len(i["hallucination_idxs"]),
        }

    # print(halc_result)
    cider_sum = 0
    chairs_sum = 0
    object_sum = 0
    meteor_sum = 0
    bleu_sum = 0
    words_sum = 0
    hallucinate_sum = 0

    hallucinate_sum_max = 2
    hallucinate_index_list = []

    for i in halc_result:
        meteor_sum += halc_result[i]["meteor"]
        bleu_sum += halc_result[i]["bleu"]
        cider_sum += halc_result[i]["cider"]
        chairs_sum += halc_result[i]["chairs"]
        object_sum += halc_result[i]["objects_num"]
        words_sum += halc_result[i]["words_num"]
        hallucinate_sum += halc_result[i]["hallucinate_num"]

    meteor_sum = meteor_sum / len(halc_result)
    log_cider_sum = cider_sum / len(halc_result)
    chairs_sum = chairs_sum / len(halc_result)
    chairi_sum = hallucinate_sum / object_sum
    bleu_sum = bleu_sum / len(halc_result)
    print("meteor: ", meteor_sum)
    print("log_cider: ", log_cider_sum)
    print("chairs: ", chairs_sum)
    print("chairi: ", chairi_sum)
    print("bleu: ", bleu_sum)
    print("hallucinate_sum: ", hallucinate_sum)


def main(args):
    # load model
    if args.voting_numbers == 1:
        settings["voting_numbers"] = [0.5]
    elif args.voting_numbers == 2:
        settings["voting_numbers"] = [0.3, 0.5]
    elif args.voting_numbers == 3:
        pass
    elif args.voting_numbers == 4:
        settings["voting_numbers"] = [0.1, 0.3, 0.5, 0.7]
    elif args.voting_numbers == 5:
        settings["voting_numbers"] = [0.1, 0.3, 0.5, 0.7, 0.9]
    else:
        print(
            "unsupport voting number, this should be from 1 to 5 and will be set to 3 by default"
        )
    settings["use_avg"] = args.avg
    model_path = args.model_path
    if args.model == "instructblip" or args.model == "llava-1.5":
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(model_path)

    # device = f"cuda:{args.gpu_id}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("preparing generation")
    if args.model == "llava-1.5":
        if args.original is True:
            print("generating original")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = CustomLlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
    elif args.model == "instructblip":
        if args.original is True:
            print("generating original")
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
            )
        else:
            model = CustomInstructBlipForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
            )
    elif args.model == "llava-next":
        if args.original is True or args.opera is True:
            print("generating original")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            model = CustomLlavaNextForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )
    # COCO dataset
    coco, coco_anns = load_coco_data(args.coco_data_dir)
    img_ids = coco.getImgIds()
    # ---------begin prepare sample dataset---------
    # Assuming coco and coco_anns are already loaded as in your original script
    img_ids = coco.getImgIds()
    sampled_img_ids = None
    if args.use_prev_sample is not None:
        # Load sampled IDs from sample.log
        with open(args.sample_save_name, "r") as f:
            sampled_img_ids = [int(line.strip()) for line in f.readlines()]

        print(f"Loaded {len(sampled_img_ids)} image IDs from {args.sample_save_name}")
    else:
        # Number of samples
        num_samples = args.image_numbers
        if args.seed is not None:
            seed(args.seed)
        # Randomly sample 500 unique image IDs
        sampled_img_ids = sample(img_ids, num_samples)

        # Write sampled IDs to a log file
        with open(args.sample_save_name, "w") as f:
            for img_id in sampled_img_ids:
                f.write(f"{img_id}\n")

        print(
            f"Sampled {num_samples} image IDs and saved them to {args.sample_save_name}"
        )
    img_files = []
    for cur_img_id in sampled_img_ids:
        cur_img = coco.loadImgs(cur_img_id)[0]
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
    base_dir = os.path.join(args.output_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # ENDCOCOTEST_LOADINGINDEX
    now = datetime.now()
    t = now.strftime("%m%d%H%M")
    filename = args.method + t + ".json"
    # ---------end prepare output data dir---------
    for idx, img_id in tqdm(enumerate(range(len(img_files))), total=len(img_files)):

        img_file = img_files[img_id]
        img_id = int(img_file.split(".jpg")[0][-6:])

        img_info = img_dict[img_id]
        assert img_info["name"] == img_file
        img_anns = set(img_info["anns"])
        img_save = {}
        img_save["image_id"] = img_id
        # begin process input data
        image_path = os.path.join(args.coco_data_dir, "val2014", img_file)
        image = load_image(image_path)
        prompt = prompt_dict[args.model]
        if args.model == "instructblip":
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                device
            )
        else:
            inputs = processor(prompt, image, return_tensors="pt").to(device)

        if args.original is True:
            if args.num_beams is not None:
                num_beams = args.num_beams
            else:
                num_beams = 1

            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=num_beams,
            )
        elif args.opera is True:
            if args.num_beams is not None:
                num_beams = args.num_beams
            else:
                num_beams = 3

            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                output_attentions=True,
                output_hidden_states=True,
                num_beams=num_beams,
                opera_decoding=True,
                scale_factor=5,
                threshold=15,
                num_attn_candidates=1,
                penalty_weights=1,
            )
        else:
            if args.num_beams is not None:
                num_beams = args.num_beams
            else:
                num_beams = 1

            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=num_beams,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        # print("decoder output ids", output_ids)
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
        if args.model == "llava-1.5":
            output_text = output_text[0].split("ASSISTANT:", 1)[-1].strip()
        elif args.model == "instructblip":
            output_text = output_text[0].strip()
        elif args.model == "llava-next":
            output_text = output_text[0].split("[/INST]", 1)[-1].strip()
        print(output_text)
        sentence_list = output_text.split(".")
        sentence_filter_list = []
        for sentence in sentence_list:
            if "unk" not in sentence:
                sentence_filter_list.append(sentence)
        output_text = ".".join(sentence_filter_list)
        # print("decoder output text", output_text)
        img_save["caption"] = output_text
        # print("image_path: ", image_path)
        # print("caption: ", output_text)
        # 获取时间

        generated_captions_path = os.path.join(base_dir, filename)
        # print("generated_captions_path", generated_captions_path)
        with open(generated_captions_path, "a") as f:
            json.dump(img_save, f)
            f.write("\n")

    print("the result is saved into", base_dir, filename)
    # -------- begin json data eval --------
    loaded_json = []

    generated_captions_path = args.output_dir
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

    # print("loaded_json:", len(loaded_json))

    # construct output file as input to CHAIR evaluation
    # output format follows https://github.com/ruotianluo/self-critical.pytorch
    formulated_output_dict = {}
    # overall result
    all_overall_scores = defaultdict(list)
    # imgToEval per image result
    img_to_eval_dict = {}
    # to save memory, load 100 captions at a time
    for start_idx in tqdm(
        range(0, len(loaded_json), 100), desc="Generating CHAIR Input"
    ):
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
    formulated_output_path = "./vlm_results/"
    if not os.path.exists(formulated_output_path):
        os.makedirs(formulated_output_path)
    formulated_output_path = formulated_output_path + filename

    with open(formulated_output_path, "w") as f:
        json.dump(formulated_output_dict, f)

    print("output file saved at: ", formulated_output_path)

    # -------- start chair eval --------
    data_dir = args.coco_data_dir
    chair_input_path = formulated_output_path
    method = args.method
    if not os.path.exists("./results"):
        os.makedirs("./results")
    chair_eval(
        chair_input_path=chair_input_path,
        model_type=args.model,
        num_images=500,
        output_dir="./results",
        dataset_name="coco",
        data_dir=data_dir,
        metric=method,
        verbosity=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="None")
    parser.add_argument("--use-prev-sample", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--original", type=bool, default=False)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--sample-save-name", type=str, default="sample.log")
    parser.add_argument("--image-numbers", type=int, default=500)
    # parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--coco-data-dir", required=True, type=str, default=None)
    parser.add_argument("--model-path", required=True, type=str, default=None)
    parser.add_argument("--avg", type=bool, default=False)
    parser.add_argument("--voting-numbers", type=int, default=3)
    parser.add_argument("--opera", type=bool, default=False)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()
    main(args)
