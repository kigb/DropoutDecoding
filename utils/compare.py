

import json


# 定义读取JSON文件的函数
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


file1 = load_json('/home/fyx/hallucination/results/org0911/llava_llava/coco/llava_llava_coco_num_images_200_chair_results.json')
file2 = load_json('/home/fyx/hallucination/results/0911maxvote4/llava_llava/coco/llava_llava_coco_num_images_200_chair_results.json')



# 获取所有的句子
sentences1 = file1.get('sentences', [])
sentences2 = file2.get('sentences', [])

# 创建image_id对应的句子字典，包含caption和mscoco_hallucinated_words
data1_dict = {sentence['image_id']: sentence for sentence in sentences1}
data2_dict = {sentence['image_id']: sentence for sentence in sentences2}

# 打开文件用于写入结果
with open('./result.log', 'w') as log_file:
    # 对比两个文件，找出第一个文件中mscoco_hallucinated_words非空而第二个文件中mscoco_hallucinated_words为空的image_id
    for image_id, data1 in data1_dict.items():
        hallucinated_words1 = data1['mscoco_hallucinated_words']
        data2 = data2_dict.get(image_id, {})
        hallucinated_words2 = data2.get('mscoco_hallucinated_words', [])

        # 判断条件：第一个文件有mscoco_hallucinated_words，第二个文件为空
        if hallucinated_words1 and not hallucinated_words2:
            # 获取两个文件的caption
            caption1 = data1['caption']
            caption2 = data2.get('caption', 'No caption available in file 2')

            # 将信息写入日志文件
            log_file.write(f"Image ID: {image_id}\n")
            log_file.write(f"Caption (File 1): {caption1}\n")
            log_file.write(f"Caption (File 2): {caption2}\n")
            log_file.write(f"Hallucinated Words (File 1): {hallucinated_words1}\n")
            log_file.write(f"Hallucinated Words (File 2): {hallucinated_words2} (Empty)\n")
            log_file.write("-" * 50 + "\n")

print("Comparison result has been saved to ./result.log")