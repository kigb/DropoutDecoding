import copy
import json

from transformers import LlavaForConditionalGeneration, LlavaProcessor
import torch
from typing import List, Optional, Tuple, Union
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
import torch.nn.functional as F
import math
from collections import Counter
from models.config import settings

import numpy as np
# seed = 24
# seed = 650
seed = 24
# seed = 5217
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def select_by_vote(outputs_all):
    id_counter = Counter()  # 用来统计每个token id出现的次数

    # 统计每个输出中最后一个token的id
    for output in outputs_all:
        token_id = torch.argmax(output[0][-1]).item()  # 获取每个output中最后一个token的id
        id_counter[token_id] += 1  # 统计该id出现次数
    most_common_id = id_counter.most_common(1)[0][0]  # 取出现次数最多的id

    for index, output in enumerate(outputs_all):
        if torch.argmax(output[0][-1]).item() == most_common_id:
            return output, index

    # If no match is found, return None
    return None, None
def select_by_average(outputs_all):
    # Use the structure of the first output as a base
    averaged_output = outputs_all[0]
    
    # Stack last tokens across outputs, transfer to CPU for NumPy operations if needed
    last_tokens = torch.stack([output[0][-1] for output in outputs_all])
    if last_tokens.device != torch.device("cpu"):
        last_tokens = last_tokens.cpu()
    
    # Compute the average with NumPy
    averaged_last_token = torch.tensor(last_tokens.numpy().mean(axis=0)).to(outputs_all[0][0].device)
    
    # Replace the last token in the first output with the averaged token
    averaged_output[0][-1] = averaged_last_token
    
    return averaged_output

class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.start_generation_pos = 0  # 用于记录生成的起始位置，在mask的时候
        self.start_image_pos = []  # 用于记录图片起始位置
        self.end_image_pos = []  # 用于记录图片结束位置
        self.is_first_generation = False  # 用于记录是否是第一次生成
        self.image_features = None  # 用于记录图片特征
        self.logits_mask_prob = []  # 用于text部分mask的概率
        self.all_outputs = []  # 用于记录所有输出，用于case study
        self.int_array = []
        # self.processor = LlavaProcessor.from_pretrained("/home/mingzhi/bridge_ziran/output/llava-v")
        # self.processor = LlavaProcessor.from_pretrained("/data3/fyx/llava-1.5-7b-hf")
        self.token_entropies = []
        self.token_ventropies = []
        self.vision_uncert_dict = None
        self.ag_mask_ids = None
        self.image_logits = None
        self.masked_numbers = []

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        self.start_image_pos.append(
            torch.where(input_ids[0] == self.config.image_token_index)[0].item())  # get start_image_pos
        num_images, num_image_patches, embed_dim = image_features.shape
        self.end_image_pos.append(self.start_image_pos[0] + num_image_patches - 1)
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[int] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # voting_numbers: Optional[int] = 3,
            # **kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        # print("settings", settings)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if input_ids.shape[-1] != 1:
            self.is_first_generation = True
            self.logits_mask_prob = []  # restore logits mask prob
            self.start_image_pos = []  # restore start image pos
            self.end_image_pos = []  # restore end image pos
            self.token_ventropies = []
            self.token_entropies = []
        else:
            self.is_first_generation = False
        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        if self.is_first_generation:
            self.start_generation_pos = attention_mask.shape[-1]  # 开始生成的位置

        def restore_attention_mask(attention_mask):
            attention_mask[:, :] = 1
            return attention_mask

        original_past_key_values = copy.deepcopy(past_key_values)
        outputs_all = []
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        outputs_all.append(outputs)
        logits = outputs[0]
        if self.is_first_generation:
            self.int_array = []
            self.ag_mask_ids = None
            # 第一次生成，需要记录image feature
            self.image_features = self.get_image_features(self.start_image_pos, self.end_image_pos, outputs)
            vision_token_logits = self.get_image_logits(logits=logits, start_image_pos=self.start_image_pos[0],
                                                        end_image_pos=self.end_image_pos[0])
            self.image_logits = vision_token_logits
            self.vision_uncert_dict = self.calculate_vision_uncertainty(vision_token_logits)



        # ------------------------------------------------------------

        # ------------------------------------------------------------
        #     with open("/home/fyx/hallucination/image_features_out.json", "w") as f:
        #         for i in range(self.image_features[1].shape[1]):
        #             # Decode the i-th token.
        #             token = self.processor.batch_decode(self.image_features[1][0][i].unsqueeze(0))
        #             # Get the corresponding variance value.
        #             variance_value = self.vision_uncert_dict["variance_per_token"][0][i]
        #             # Get other fields from vision_uncert_dict.
        #             epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0][i]
        #             alea_uncert = self.vision_uncert_dict["alea_uncert_per_token"][0][i]
        #
        #             # Write the data to the file.
        #             f.write(
        #                 f"Token {i + 1}: {token}, Variance: {variance_value}, Epis_Uncert: {epis_uncert}, Alea_Uncert: {alea_uncert}\n")
        loss = None
        max_vote = True
        if not self.is_first_generation:
        # if True:
            self.masked_numbers = []
            outputs_all = []
            probs = settings['voting_numbers']
            method_ = "logits"
            for mprob in probs:
                original_past_key_values_ = copy.deepcopy(original_past_key_values)
                # attention_mask = restore_attention_mask(attention_mask)
                attention_mask = self.get_image_attention_mask(logits, attention_mask, method="epis",
                                                               prob=mprob)
                # original "table", image features seach table image 1 projection cat dog image 2 project table desk
                # img token 1 2 3 4 5 3 has table, 1 2 4 5 mprob 0.3 0.5
                # attention_mask = self.get_image_attention_mask(logits, attention_mask, method=method_)
                outputs_all.append(self.language_model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=original_past_key_values_,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                ))

            outputs_r,index = select_by_vote(outputs_all)
                # print("avg")
                # with open("/home/fyx/hallucination/tmp/llava_masked_numbers.log", "a") as file:
                #     file.write(f"{self.masked_numbers[index]},")

            # self.logits_mask_prob.append(1 / torch.max(outputs_r[0][-1]).item())
            # en, ven = self.calculate_entropy_varentropy(outputs_r[0][0][-1])
            # self.token_entropies.append(en)
            # self.token_ventropies.append(ven)
            return LlavaCausalLMOutputWithPast(
                loss=loss,
                logits=outputs_r[0],
                past_key_values=outputs_r.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # self.logits_mask_prob.append(1 / torch.max(logits[-1][-1]).item())
        # en, ven = self.calculate_entropy_varentropy(logits[0][-1])
        # self.token_entropies.append(en)
        # self.token_ventropies.append(ven)
        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_image_features(self, start_image_pos, end_image_pos, outputs):
        """
        get image features

        Args:
            start_image_pos: start position of image token
            end_image_pos: end position of image token
            outputs: model outputs

        Returns:
            image_features: image features
        """
        if len(start_image_pos) == 0 or len(end_image_pos) == 0:
            return None
        logits = outputs['logits']
        # print(start_image_pos, end_image_pos)
        image_logits = self.get_image_logits(logits=logits, start_image_pos=start_image_pos[0],
                                             end_image_pos=end_image_pos[0])
        values, ids = self.get_topk_token_id(image_logits, topk=5)
        image_features = (values, ids)
        return image_features

    def get_image_logits(self, logits, start_image_pos, end_image_pos):
        """
        get image token logits

        Args:
            logits: model outputs['logits']
            start_image_pos: start position of image token
            end_image_pos: end position of image token

        Returns:
            image_projected_features: image token logits
        """
        # 最后一层hiddenstate中image token的特征
        image_projected_features = logits[:, start_image_pos:end_image_pos + 1]
        return image_projected_features

    def get_topk_token_id(self, image_logits, topk=5):
        """
        get value,token id of topk similarity

        Args:
            image_logits: image token logits
            topk: topk k

        Returns:
            topk_values: topk values of image token logits
            topk_indices: topk indices of image token logits
        """
        topk_values, topk_indices = torch.topk(image_logits, topk, dim=-1)
        return topk_values, topk_indices

    def get_overlap_image_tokens(self, logits, id=-1):
        """
        get overlap image tokens

        Args:
            logits: model outputs['logits']
            id: max token id
            image token project topk tokens -> generation token overlap, if one image token
            has overlap with generation token, then keep it

        Returns:
            overlap_image_tokens: overlap image tokens
        """
        if id != -1:
            max_ids = torch.tensor([[id]], device='cuda:2')
        else:
            max_ids = torch.argmax(logits, dim=-1)  # [batch_size, sequence_length]

        top5_ids = self.image_features[1]  # [batch_size, 1948, 5]
        # 假设 logits 是 [batch_size, sequence_length] 的张量
        # 假设 top5_ids 是 [batch_size, 1948, 5] 的张量

        # 获取第一个样本的 max_ids 和 top5_ids
        max_ids_sample = max_ids[0]  # 形状为 [sequence_length]
        top5_ids_sample = top5_ids[0]  # 形状为 [1948, 5]

        # 比较 top5_ids_sample 中每一行是否包含 max_ids_sample
        matches = (top5_ids_sample == max_ids_sample)  # 形状为 [3, 5]

        # 在每一行的维度上进行 logical OR 操作，如果某行包含 max_ids_sample，那么该行的结果为 True
        matched_rows = matches.any(dim=1)  # 形状为 [3]

        # 找到 matched_rows 中为 True 的行的索引
        matched_indices = torch.nonzero(matched_rows).squeeze()

        offset = self.start_image_pos[0]
        adjusted_indices = matched_indices + offset

        # 返回调整后的索引
        return adjusted_indices

    def get_image_attention_mask(self, logits, attention_mask, method="keep_overlap", prob=0.5, ids=[]):
        """
        get image attention mask

        Args:
            logits: model outputs['logits']
            method: method to get image attention mask

        Returns:
            image_attention_mask: image attention mask
        """
        if method == "overlap":
            matched_indices = self.get_overlap_image_tokens(logits)
            adjusted_indices_tensor = torch.tensor(matched_indices)
            # 更新 attention_mask，设定这些位置的值为 0
            attention_mask[:, adjusted_indices_tensor] = 0
        elif method == "keep_overlap":
            matched_indices = self.get_overlap_image_tokens(logits)
            adjusted_indices_tensor = matched_indices.clone().detach()

            indices = torch.arange(self.start_image_pos[0], self.end_image_pos[0] + 1)

            # 确保索引范围不超过 attention_mask 的维度
            indices = indices[indices < attention_mask.shape[-1]]

            # 生成随机数 tensor
            random_nums = torch.rand(indices.shape[0])

            mask_condition = random_nums < prob

            # 应用掩码修改
            attention_mask[:, indices[mask_condition]] = 0
            if adjusted_indices_tensor.shape == 0:
                return attention_mask
            attention_mask[:, adjusted_indices_tensor] = 1
        elif method == "VQA":
            for id in ids:
                matched_indices = self.get_overlap_image_tokens(logits, id)
                adjusted_indices_tensor = torch.tensor(matched_indices)

                indices = torch.arange(self.start_image_pos[0], self.end_image_pos[0] + 1)

                # 确保索引范围不超过 attention_mask 的维度
                indices = indices[indices < attention_mask.shape[-1]]

                # 生成随机数 tensor
                random_nums = torch.rand(indices.shape[0])

                mask_condition = random_nums < prob

                # 应用掩码修改
                attention_mask[:, indices[mask_condition]] = 0
                if adjusted_indices_tensor.shape == 0:
                    continue
                attention_mask[:, adjusted_indices_tensor] = 1

        elif method == "all_image":
            attention_mask[:, self.start_image_pos[0]:self.end_image_pos[0] + 1] = 0
        elif method == "random_image":
            # mask based on random number
            for i in range(self.start_image_pos[0], self.end_image_pos[0] + 1):
                random_num = torch.rand(1)
                if random_num < prob:
                    attention_mask[:, i] = 0
        elif method == "logits":
            # mask based on self.logits_mask_prob
            for i in range(len(self.logits_mask_prob)):
                # create a random number between 0 and 1
                if i + self.start_generation_pos >= attention_mask.shape[-1]:
                    break
                random_num = torch.rand(1)
                if random_num < self.logits_mask_prob[i]:
                    attention_mask[:, i + self.start_generation_pos] = 0
            attention_mask[:, -3:] = 1
        elif method == "entropy":
            for i in range(len(self.token_entropies)):
                if self.token_entropies[i] < 0.1 and self.token_ventropies[i] < 0.1:
                    attention_mask[:, i + self.start_generation_pos] = 1
                elif self.token_entropies[i] > 5 and self.token_ventropies[i] > 5:
                    random_num = torch.rand(1)
                    attention_mask[:, i + self.start_generation_pos] = (random_num > 0.5)
                else:
                    random_num = torch.rand(1)
                    if random_num < self.logits_mask_prob[i]:
                        attention_mask[:, i + self.start_generation_pos] = 0
            attention_mask[:, -3:] = 1
        elif method == "agressive":
            # agressively mask image token at a prob
            # 确定需要掩码的范围和数量
            if self.ag_mask_ids != None:
                attention_mask[:, self.ag_mask_ids] = 0
                return
            start_pos = self.start_image_pos[0]
            end_pos = self.end_image_pos[0]
            num_tokens = end_pos - start_pos  # 计算需要掩码的 token 数量
            num_mask_tokens = int(prob * num_tokens)  # 30%的 token 数量

            # 获取要掩码范围内的索引
            all_indices = torch.arange(start_pos, end_pos)

            # 随机选择 num_mask_tokens 个索引进行掩码
            masked_indices = torch.randperm(num_tokens)[:num_mask_tokens] + start_pos
            self.ag_mask_ids = masked_indices
            # 将这些索引的位置在 attention_mask 中设为 0
            attention_mask[:, masked_indices] = 0
        elif method== "epis":
            # matched_indices = self.get_overlap_image_tokens(logits)
            # adjusted_indices_tensor = matched_indices.clone().detach()
            # epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"]
            # indexes = torch.where(epis_uncert[0] < 1)[0]
            # random_tensor = torch.rand_like(indexes.float())
            # # depend on epis score mask least percent
            # # uniform
            # # compare mask method whatever performance
            # mask = random_tensor <= prob
            # filtered_indexes = indexes[mask]
            # attention_mask[:,filtered_indexes+self.start_image_pos[0]] = 0
            # attention_mask[:, adjusted_indices_tensor] = 1

            matched_indices = self.get_overlap_image_tokens(logits)
            adjusted_indices_tensor = matched_indices.clone().detach()

            # epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]
            #
            # # 1. 计算 `epis_uncert` 的 `prob` 百分位数阈值
            # threshold = torch.quantile(epis_uncert, prob)
            #
            # # 2. 获取小于或等于该阈值的索引
            # indexes = torch.where(epis_uncert >= threshold)[0]
            #
            # # 3. 设置 `attention_mask` 中 `filtered_indexes` 的位置为 0
            # attention_mask[:, indexes + self.start_image_pos[0]] = 0
            # attention_mask[:, adjusted_indices_tensor] = 1

            # epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]
            #
            # # 2. Normalize epis_uncert values to the range [0.1, 0.7]
            # min_value = epis_uncert.min()
            # max_value = epis_uncert.max()
            # normalized_probs = 0.1 + 0.6 * (epis_uncert - min_value) / (max_value - min_value)  # Scales to [0.1, 0.7]
            #
            # # 3. Generate a random tensor with the same shape as `epis_uncert`
            # random_tensor = torch.rand_like(epis_uncert)
            #
            # # 4. Determine the indices where the random values are less than the normalized probabilities
            # mask = random_tensor < normalized_probs
            # filtered_indexes = torch.where(mask)[0]
            #
            # # 5. Set `attention_mask` at these filtered indices to 0
            # attention_mask[:, filtered_indexes + self.start_image_pos[0]] = 0
            #
            # # 6. Restore attention for matched indices to 1
            # attention_mask[:, adjusted_indices_tensor] = 1
            # 假设 epis_uncert 是你的输入张量
            epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]

            # 获取 epis_uncert 的 0.1 和 0.7 分位数
            q_low = torch.quantile(epis_uncert, 0)
            q_high = torch.quantile(epis_uncert, 1)

            # 使用线性插值将 epis_uncert 映射到 [0.1, 0.7] 范围
            # clamp 是为了确保数据在分位数范围内，并进行插值
            normalized_probs = 0.1 + (prob - 0.1) * (epis_uncert.clamp(min=q_low, max=q_high) - q_low) / (
                        q_high - q_low)

            # 生成与 epis_uncert 相同形状的随机张量
            random_tensor = torch.rand_like(epis_uncert)

            # 根据生成的随机张量与 normalized_probs 确定 mask
            mask = random_tensor < normalized_probs
            filtered_indexes = torch.where(mask)[0]
            
            # 设置 attention_mask 中的指定位置为 0
            attention_mask[:, filtered_indexes + self.start_image_pos[0]] = 0

            # 恢复 attention_mask 的某些索引为 1
            attention_mask[:, adjusted_indices_tensor] = 1
            zero_count = (attention_mask[0] == 0).sum().item()
            self.masked_numbers.append(zero_count)
        elif method == "epis_no_overlap":
            epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]

            # 获取 epis_uncert 的 0.1 和 0.7 分位数
            q_low = torch.quantile(epis_uncert, 0)
            q_high = torch.quantile(epis_uncert, 1)

            # 使用线性插值将 epis_uncert 映射到 [0.1, 0.7] 范围
            # clamp 是为了确保数据在分位数范围内，并进行插值
            normalized_probs = 0.1 + (prob - 0.1) * (epis_uncert.clamp(min=q_low, max=q_high) - q_low) / (
                        q_high - q_low)

            # 生成与 epis_uncert 相同形状的随机张量
            random_tensor = torch.rand_like(epis_uncert)

            # 根据生成的随机张量与 normalized_probs 确定 mask
            mask = random_tensor < normalized_probs
            filtered_indexes = torch.where(mask)[0]
            
            # 设置 attention_mask 中的指定位置为 0
            attention_mask[:, filtered_indexes + self.start_image_pos[0]] = 0

        return attention_mask

    def calculate_entropy_varentropy(self, logits):
        """
        Calculate the entropy and varentropy of the probability distribution using log_softmax.

        Args:
            logits (torch.Tensor): Input tensor of logits with shape [vocab_size].

        Returns:
            entropy (float): The calculated entropy.
            varentropy (float): The calculated varentropy.
        """
        # Calculate log probabilities using log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Calculate entropy in base-2
        entropy = -torch.sum(probs * log_probs) / math.log(2)

        # Calculate varentropy
        varentropy = torch.sum(probs * (log_probs / math.log(2) + entropy) ** 2)

        return entropy.item(), varentropy.item()

    def calculate_vision_uncertainty(self, logits):
        """
        Calculate various metrics for a distribution over tokens (Specificity, only vision tokens).

        Args:
        logits (torch.Tensor): Input tensor of shape [B, L_vision, V] where B is batch size,
                            L_vision is sequence length of vision tokens, and V is vocabulary size.

        Returns:
        dict: A dictionary containing the calculated metrics.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # [B, L, V]

        # 1. Expected value for each position
        expected_values = torch.mean(probs, dim=-1)  # [B, L]

        # 2. Variance for each position
        variance_per_token = torch.var(probs, dim=-1)  # [B, L]
        variance = torch.mean(variance_per_token, dim=-1)

        # 3. Average distribution over V
        p_avg = torch.mean(probs, dim=1)  # [B, V]

        # 4. Positional uncertainty (Epistemic)
        epi_per_token = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(p_avg.unsqueeze(1) + 1e-10)),
                                  dim=-1)  # [B, L]

        # 5. Positional entropy (Aleatoric)
        alea_per_token = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [B, L]

        # -------------------------------
        # for traj(image)-level uncertainty, please take expectation over L_vision
        epistemic_uncertainty = torch.mean(epi_per_token, dim=-1)  # [B]
        aleatoric_uncertainty = torch.mean(alea_per_token, dim=-1)  # [B]
        # -------------------------------

        return {
            # "expected_values": expected_values,
            "variance_per_token": variance_per_token,
            # "p_avg": p_avg,
            "epis_uncert_per_token": epi_per_token,
            "alea_uncert_per_token": alea_per_token,
            "variance": variance,
            "epis_uncert": epistemic_uncertainty,
            "alea_uncert": aleatoric_uncertainty
        }

    def lowest_percent_kl_indices(self, image_logits, logits, percent=0.1):
        # self_image_logits: [1, N, M]
        # logits: [1, 1, M]

        # 去除批次维度，使 self_image_logits 维度为 [N, M]，logits 维度为 [1, M]
        self_image_logits = image_logits.squeeze(0)  # Shape: [N, M]
        logits = logits.squeeze(0)  # Shape: [1, M]

        # 计算每一个 N 个向量的 KL 散度，保留在一个形状为 [N] 的张量中
        kl_divergences = F.kl_div(F.log_softmax(self_image_logits, dim=-1),
                                  F.softmax(logits, dim=-1).expand_as(self_image_logits),
                                  reduction='none').sum(dim=-1)

        # 计算需要的最小百分比的数量
        num_lowest = int(percent * kl_divergences.numel())

        # 使用 topk 找到最小的百分比 KL 散度索引
        _, lowest_indices = torch.topk(kl_divergences, num_lowest, largest=False)

        return lowest_indices