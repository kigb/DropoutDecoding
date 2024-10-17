import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from random import sample, seed
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

# from pope_loader import POPEDataSet

from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration


from PIL import Image
from torchvision.utils import save_image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib as mpl

import json
def load_coco_data(data_dir):
    annotation_file_path = data_dir + "annotations/instances_val2014.json"
    caption_file_path = data_dir + "annotations/captions_val2014.json"
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])
    coco = COCO(caption_file_path)
    return coco, coco_anns
MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model",default="/data2/fyx/llava-v1.5-7b")
parser.add_argument("--gpu-id", type=int, help="specify the gpu to load the model.")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
         "in xxx=yyy format will be merged into config file (deprecate), "
         "change to --cfg-options instead.",
)
parser.add_argument("--data_path", type=str, default="COCO_2014/val2014/", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="num workers")
parser.add_argument("--save-name", type=str, default="output")
parser.add_argument("--use-prev-sample", type=str, default=None)
parser.add_argument("--beam", type=int)
parser.add_argument("--sample", action='store_true')
parser.add_argument("--scale_factor", type=float, default=50)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num_attn_candidates", type=int, default=5)
parser.add_argument("--penalty_weights", type=float, default=1.0)
args = parser.parse_known_args()[0]

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# ========================================
#             Model Initialization
# ========================================
print('Initializing Model')

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print(vis_processors["eval"].transform)
print("Done!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

coco, coco_anns = load_coco_data("/data3/fyx/COCO/")
img_ids = coco.getImgIds()
# ---------begin prepare sample dataset---------
# Assuming coco and coco_anns are already loaded as in your original script
img_ids = coco.getImgIds()
sampled_img_ids = None
if args.use_prev_sample is not None:
    # Load sampled IDs from sample.log
    with open('sample.log', 'r') as f:
        sampled_img_ids = [int(line.strip()) for line in f.readlines()]

    print(f"Loaded {len(sampled_img_ids)} image IDs from sample.log")
else:
    # Number of samples
    num_samples = 500
    if args.seed is not None:
        seed(args.seed)
    # Randomly sample 500 unique image IDs
    sampled_img_ids = sample(img_ids, num_samples)

    # Write sampled IDs to a log file
    with open('sample.log', 'w') as f:
        for img_id in sampled_img_ids:
            f.write(f"{img_id}\n")

    print(f"Sampled {num_samples} image IDs and saved them to sample.log")
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
base_dir = os.path.join("/home/fyx/vlm_outputs")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
# ENDCOCOTEST_LOADINGINDEX
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
    # begin process input data
    image_path = "/data3/fyx/COCO/val2014/" + img_file
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    image = image.to(device)

    qu = "Please describe this image in detail."
    template = INSTRUCTION_TEMPLATE[args.model]
    qu = template.replace("<question>", qu)

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt": qu},
                use_nucleus_sampling=args.sample,
                num_beams=args.beam,
                max_new_tokens=512,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
            )


    output_text = out[0]
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

    generated_captions_path = os.path.join(
        base_dir,
        filename)
    # print("generated_captions_path", generated_captions_path)
    with open(generated_captions_path, "a") as f:
        json.dump(img_save, f)
        f.write("\n")

print("the caption is saved into",generated_captions_path)




