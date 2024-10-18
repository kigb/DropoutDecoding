import argparse
from transformers import LlavaNextProcessor
from models.utils import CustomLlavaNextForConditionalGeneration
import torch
from PIL import Image
def main(args):
    model_path = "/data3/fyx/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path)
    device = 'cuda:1'
    model = CustomLlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map = device
    )
    image = Image.open(args.img_path)
    while True:
        img_path = input("Please input the image path, press Enter to continue.")
        if img_path != "":
            image = Image.open("/home/fyx/vlm_images/" + img_path)
        prompt = "[INST] <image>\nDescribe the image in detail? [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=512, use_input_embeddings=False,num_beams=1)
        output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
        output_text = output_text[0].split('[/INST]', 1)[-1].strip()
        print(output_text)
        input("Want to generate again? Press Enter to continue, Ctrl+C to exit.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path",type=str,default="/home/fyx/vlm_images/COCO_val2014_000000117425.jpg")
    args = parser.parse_args()
    main(args)