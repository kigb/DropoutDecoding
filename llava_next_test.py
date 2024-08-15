# Import the necessary modules from the transformers library
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches # one function explicitly used in forward.
import torch
from PIL import Image
import requests
import json
import numpy as np
from typing import List, Optional, Tuple, Union
import json
import copy


class EarlyExitException(Exception):
    def __init__(self, message, early_exit_output):
        super().__init__(message)
        self.early_exit_output = early_exit_output

# Define a new class that subclasses LlavaNextForConditionalGeneration
class CustomLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.start_generation_pos = 0 # 用于记录生成的起始位置，在mask的时候

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        feature_lens,
        inputs_embeds,
        input_ids,
        attention_mask,
        position_ids=None,
        labels=None,
        image_token_index=None,
        ignore_index=-100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_token_index = image_token_index if image_token_index is not None else self.config.image_token_index
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            # ! in llava 1.6, number of patches is variable
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True
            if batch_size > 1:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=feature_lens.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_labels = None
        if labels is not None:
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
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
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # output final_embedding, final_attention_mask, position_ids, final_labels in proper format
        # print("final_embedding.shape", final_embedding.shape)
        # print("image_to_overwrite", image_to_overwrite[0].sum())
        # apend final_embedding, image_to_overwrite to a json file
        # with open("final_embedding.json", "w") as f:
        #     json.dump(final_embedding.tolist(), f) # [B, L, 4096], dtype, 
        # with open("text_to_overwrite.json", "w") as f:
        #     json.dump(text_to_overwrite.tolist(), f)
            
        early_exit_output = {
            "final_embedding": final_embedding,
            "text_to_overwrite": text_to_overwrite,
            "image_to_overwrite": image_to_overwrite,
        }
        return final_embedding, final_attention_mask, position_ids, final_labels, early_exit_output
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
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
        use_input_embeddings = False,
    ) -> Union[dict, Tuple]:
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
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

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

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            # In case image_token_index is not in the embeddings (extra token but embedding don't have it)
            for_inputs_embeds_ids = input_ids.clone()
            for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
            inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                # ! infer image_num_patches from image_sizes
                image_num_patches = [
                    image_size_to_num_patches(
                        image_size=imsize,
                        grid_pinpoints=self.config.image_grid_pinpoints,
                        patch_size=self.config.vision_config.image_size,
                    )
                    for imsize in image_sizes
                ]
                # figure out if pixel_values is concatenated or stacked
                if pixel_values.dim() == 5:
                    # stacking when input is (batch_size, num_patches, num_channels, height, width)
                    _pixel_values_list = [
                        pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    pixel_values = torch.cat(_pixel_values_list, dim=0)
                elif pixel_values.dim() != 4:
                    # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

                image_features = self.vision_tower(pixel_values, output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                image_features = self.multi_modal_projector(selected_image_feature)

                image_features = torch.split(image_features, image_num_patches, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"

                image_features, feature_lens = self.pack_image_features(
                    image_features,
                    image_sizes,
                    image_newline=self.image_newline,
                )

                inputs_embeds = inputs_embeds.to(image_features.dtype)
                inputs_embeds, attention_mask, position_ids, labels, early_exit_output = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )

                print(f"dont use early stop exit on input embeddings")
                
                if use_input_embeddings:
                    print("Exiting early from generate()")
                    raise EarlyExitException("Early exit triggered", early_exit_output)
            
            # pixel_values is not None but is empty ---> text only cases
            elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
                # there are no images
                pass

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

        # 假设默认为use cache，这样不需要在generate函数中传入start_generation_pos
        # 暂时假设不是batch generation，后续直接修改即可
        if past_key_values is None or input_ids.shape[1] != 1:
            start_generation_pos = attention_mask.shape[-1] # 开始生成的位置

        """
            获取需要的attention_mask, 存在不同方法，以random为例
        """
        def custom_attention_mask(attention_mask, random_prob=0.1):
            # 生成随机的attention_mask，设置第二个维度从start_generation_pos开始的位置到末尾以random_prob为概率进行mask
            random_mask = torch.rand(attention_mask.shape[-1] - start_generation_pos).to(attention_mask.device) < random_prob
            attention_mask[:, start_generation_pos:] = random_mask
            return attention_mask

        """
            还原attention_mask
        """
        def restore_attention_mask(attention_mask):
            attention_mask[:, start_generation_pos:] = 1
            return attention_mask
        # kv会改变，需要深拷贝来进行储存
        original_past_key_values = copy.deepcopy(past_key_values)
        output_hidden_states = True
       
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

        # 用处理后后的attention_mask进行生成
        attention_mask = custom_attention_mask(attention_mask=attention_mask)
        outputs_random = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=original_past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits_random = outputs_random[0]
        attention_mask = restore_attention_mask(attention_mask)
        """
        # 如果不需要earlyexit，直接返回outputs，可以选择保存原始生成，即outputs的hidden states与attention，仅仅保留outputs_random的logits
        # 如下所示：
        # return LlavaNextCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits_random,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        """

        raise EarlyExitException("Early exit triggered", outputs)
        logits = outputs[0]
        return outputs
        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     if attention_mask is not None:
        #         shift_attention_mask = attention_mask[..., 1:]
        #         shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
        #         shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        #     else:
        #         shift_logits = logits[..., :-1, :].contiguous()
        #         shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        #     )

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return LlavaNextCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
import torch.nn.functional as F
def calculate_cosine_similarity(inputs_embeds, all_embeddings, batch_size=24):
    """
    计算输入嵌入与所有嵌入之间的余弦相似度。

    参数:
    - inputs_embeds: 输入嵌入，形状为 [1, L, D]
    - all_embeddings: 所有嵌入，形状为 [V, D]
    - batch_size: 批处理大小

    返回:
    - cosine_sim: 余弦相似度矩阵，形状为 [L, filtered_V]
    - original_token_ids: 过滤后的原始token ID
    """
    inputs_embeds = inputs_embeds.squeeze(0)  # 变成 [L, D]
    L, D = inputs_embeds.shape
    V = all_embeddings.shape[0]

    # 找到全零向量的索引
    zero_vector_indices = (all_embeddings == 0).all(dim=1)
    # 过滤掉全零向量
    filtered_embeddings = all_embeddings[~zero_vector_indices]
    filtered_V = filtered_embeddings.shape[0]

    # 创建一个空的张量来存储结果
    cosine_sim = torch.empty((L, filtered_V), device=inputs_embeds.device)

    # 分批处理
    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        inputs_batch = inputs_embeds[start:end]  # 形状为 [batch_size, D]

        # 计算当前批次与 filtered_embeddings 的余弦相似度
        batch_cosine_sim = F.cosine_similarity(
            inputs_batch.unsqueeze(1),  # 变成 [batch_size, 1, D]
            filtered_embeddings.unsqueeze(0),  # 变成 [1, filtered_V, D]
            dim=-1
        )  # 得到的形状为 [batch_size, filtered_V]

        # 将结果存储在预先分配的张量中
        cosine_sim[start:end] = batch_cosine_sim

    # 映射回原始的 token ids
    original_token_ids = torch.arange(V, device=inputs_embeds.device)[~zero_vector_indices]

    return cosine_sim, original_token_ids

import json
def interpret_top_k_tokens(name, cosine_sim, original_token_ids, processor, k=5):
    """
    解释得到的Top-K token，并记录对应的余弦相似度值。

    参数:
    - cosine_sim: 余弦相似度矩阵，形状为 [L, filtered_V]
    - original_token_ids: 过滤后的原始token ID
    - processor: 包含tokenizer的处理器
    - k: Top-K的值

    返回:
    - sentences: 解释后的句子列表
    """
    # 获取最相似的 token ids 和其对应的余弦相似度值的前 K 个
    top_k_values, top_k_indices = torch.topk(cosine_sim, k, dim=-1)

    # 映射回原始的 token ids
    top_k_token_ids = original_token_ids[top_k_indices]

    # 将 token ids 和对应的余弦相似度值转换为字符串并记录
    top_k_tokens_with_scores = [
        [{"token": processor.tokenizer.convert_ids_to_tokens([token_id.item()])[0], "cosine_similarity": value.item()}
         for token_id, value in zip(token_ids, values)]
        for token_ids, values in zip(top_k_token_ids, top_k_values)
    ]

    # 将结果保存到JSON文件中
    with open(name, "w") as f:
        json.dump(top_k_tokens_with_scores, f, indent=2)

    # # 转换为句子
    # sentences = [processor.tokenizer.convert_tokens_to_string(
    #     [item["token"] for item in token_list]) for token_list in top_k_tokens_with_scores]

    # return sentences
    return

def output_to_top_k_tokens(outputs, tokenizer, file_name, k=5):
    logits = outputs['logits']  # torch.Size([1, 1044, 32064])

    # Step 1: 对 logits 进行 softmax 操作
    probs = F.softmax(logits, dim=-1)  # torch.Size([1, 1044, 32064])

    # Step 2: 使用 torch.topk 提取每个位置上的 top-k tokens 及其概率
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)  # top_k_probs: torch.Size([1, 1044, 5]), top_k_indices: torch.Size([1, 1044, 5])

    # Step 3: 使用 tokenizer 将 token IDs 转换为文本
    batch_size, seq_len, _ = top_k_probs.size()
    results = []

    for i in range(batch_size):
        seq_results = []
        for j in range(seq_len):
            token_results = []
            token_texts = tokenizer.convert_ids_to_tokens(top_k_indices[i, j].tolist())
            for l in range(k):
                token_results.append({
                    "token": token_texts[l],
                    "prob": top_k_probs[i, j, l].item()
                })
            seq_results.append(token_results)
        results.append(seq_results)

    # 将结果写入文件
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



def inspect_decode(all_embeddings, processor, early_exit_output):
    final_embedding = early_exit_output['final_embedding'] # [B, L, D]
    text_to_overwrite = early_exit_output['text_to_overwrite']
    image_to_overwrite = early_exit_output['image_to_overwrite']
    inputs_embeds = final_embedding

    # 假设 inputs_embeds 的形状为 [1, L, D]，all_embeddings 的形状为 [V, D]
    inputs_embeds = inputs_embeds.squeeze(0)  # 变成 [L, D]
    L, D = inputs_embeds.shape
    V = all_embeddings.shape[0]

    # 找到全零向量的索引
    zero_vector_indices = (all_embeddings == 0).all(dim=1)
    # 过滤掉全零向量
    filtered_embeddings = all_embeddings[~zero_vector_indices]
    filtered_V = filtered_embeddings.shape[0]

    # 设置一个批处理大小，以避免 CUDA OOM
    batch_size = 24

    # 创建一个空的张量来存储结果
    cosine_sim = torch.empty((L, filtered_V), device=inputs_embeds.device)

    # 分批处理
    for start in range(0, L, batch_size):
        end = min(start + batch_size, L)
        inputs_batch = inputs_embeds[start:end]  # 形状为 [batch_size, D]

        # 计算当前批次与 filtered_embeddings 的余弦相似度
        batch_cosine_sim = F.cosine_similarity(
            inputs_batch.unsqueeze(1),  # 变成 [batch_size, 1, D]
            filtered_embeddings.unsqueeze(0),  # 变成 [1, filtered_V, D]
            dim=-1
        )  # 得到的形状为 [batch_size, filtered_V]

        # 将结果存储在预先分配的张量中
        cosine_sim[start:end] = batch_cosine_sim

    # 获取最相似的 token ids
    filtered_token_ids = torch.argmax(cosine_sim, dim=-1)

    # 映射回原始的 token ids
    original_token_ids = torch.arange(V, device=inputs_embeds.device)[~zero_vector_indices]
    token_ids = original_token_ids[filtered_token_ids]

    # 将 token ids 转换为字符串
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids.squeeze().tolist())
    sentence = processor.tokenizer.convert_tokens_to_string(tokens)

    print(sentence)
    return sentence

import matplotlib.pyplot as plt
import numpy as np
import os

def saveimg(image, variable_name):
    # 生成保存路径
    save_path = os.path.join('tmp', f'{variable_name}.png')
    
    # 保存图片
    plt.imsave(save_path, image)

if __name__ == "__main__":
    # model_path = '/data/ziran/hf_cache/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f'

    # model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/data3/fyx/llava-v1.6-mistral-7b-hf"
    processor = LlavaNextProcessor.from_pretrained(model_path)

    device = 'cuda:3'
    model = CustomLlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map = device
        # low_cpu_mem_usage=True,
        # load_in_4bit=True
    )


    # use_input_embeddings = True
    use_input_embeddings = False
    # prepare image and text prompt, using the appropriate prompt template
    # url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # image = Image.open(requests.get(url, stream=True).raw)
    # index = 0
    for index in range(1):
        img_path = f"/home/fyx/vlm_images/COCO_val2014_000000012443.jpg"
        image = Image.open(img_path)
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

        # input_ids = processor.tokenizer(prompt, return_tensors="pt",)['input_ids'].to(device)
        # for_inputs_embeds_ids = input_ids.clone()
        # for_inputs_embeds_ids[(input_ids == model.config.image_token_index)] = 0
        # inputs_embeds = model.get_input_embeddings()(for_inputs_embeds_ids)

        # ------------------------------------
        globals()['processor'] = processor
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        try:
            early_exit_output = model.generate(**inputs, max_new_tokens=100, use_input_embeddings=use_input_embeddings)
        except EarlyExitException as e:
            early_exit_output = e.early_exit_output
        # ------------------------------------

        
        if use_input_embeddings:
            # ------------------------------------
            file_path = "early_exit_output.pth"
            torch.save(early_exit_output, file_path)
            early_exit_output = torch.load(file_path)
            # ------------------------------------
            embedding_layer = model.get_input_embeddings()
            all_embeddings = embedding_layer.weight.data
            file_path = "input_all_embeddings.pth"
            torch.save(all_embeddings, file_path)
            all_embeddings = torch.load(file_path)
            # ------------------------------------
            
            # # ------------------------------------
            # file_path = "early_exit_output.pth"
            # early_exit_output = torch.load(file_path)
            # file_path = "input_all_embeddings.pth"
            # all_embeddings = torch.load(file_path)
            # # ------------------------------------
            # # inspect_decode(all_embeddings, processor, early_exit_output)

            # ------------------------------------
            final_embedding = early_exit_output['final_embedding'] # [B, L, D]
            text_to_overwrite = early_exit_output['text_to_overwrite']
            image_to_overwrite = early_exit_output['image_to_overwrite']
            inputs_embeds = final_embedding
            cosine_sim, original_token_ids = calculate_cosine_similarity(inputs_embeds, all_embeddings, batch_size=24)
            save_name = f'./output/input_{index}_top_k_tokens_with_scores.json'
            sentences = interpret_top_k_tokens(save_name, cosine_sim, original_token_ids, processor, k=20)
        else:
            logits = early_exit_output['logits'] # torch.Size([1, 1044, 32064])
            logits_tensor = early_exit_output['logits'].cpu()
            last_hidden_states = early_exit_output['hidden_states'][-1].cpu()
            torch.save(logits_tensor, 'out_logits_tensor.pt')
            torch.save(last_hidden_states, 'last_hidden_states.pt')
            output_to_top_k_tokens(early_exit_output, processor.tokenizer, f'./output/output_{index}_top_k_tokens_with_scores.json', k=20)

    # print(processor.decode(output[0], skip_special_tokens=True))
