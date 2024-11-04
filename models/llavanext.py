# Import the necessary modules from the transformers library
import math

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.models.llava_next.modeling_llava_next import image_size_to_num_patches, LlavaNextCausalLMOutputWithPast # one function explicitly used in forward.
import torch
from PIL import Image
import requests
import json
import numpy as np
from typing import List, Optional, Tuple, Union
import json
import copy
from collections import Counter
import torch.nn.functional as F
# seed = 114115
seed = 506
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def argmax(output):
    return torch.argmax(output).item()

def select_by_vote(outputs_all):
    id_counter = Counter()  # 用来统计每个token id出现的次数
    
    # 统计每个输出中最后一个token的id
    for output in outputs_all:
        token_id = argmax(output[0][-1])  # 获取每个output中最后一个token的id
        id_counter[token_id] += 1  # 统计该id出现次数
    most_common_id = id_counter.most_common(1)[0][0]  # 取出现次数最多的id
    
    for output in outputs_all:
        if argmax(output[0][-1]) == most_common_id:
            return output
    
    return None  # 如果没有找到，返回None


class EarlyExitException(Exception):
    def __init__(self, message, early_exit_output):
        super().__init__(message)
        self.early_exit_output = early_exit_output

# Define a new class that subclasses LlavaNextForConditionalGeneration
class CustomLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.start_generation_pos = 0 # 用于记录生成的起始位置，在mask的时候
        self.start_image_pos = [] # 用于记录图片起始位置
        self.end_image_pos = [] # 用于记录图片结束位置
        self.is_first_generation = False # 用于记录是否是第一次生成
        self.image_features = None # 用于记录图片特征
        self.logits_mask_prob = [] # 用于text部分mask的概率
        self.all_outputs = [] # 用于记录所有输出，用于case study
        # self.processor = LlavaNextProcessor.from_pretrained("/data3/fyx/llava-v1.6-mistral-7b-hf")
        self.do_step = True # 用于控制是否进行step操作
        self.skip_steps = 0 # 用于控制跳过的step数
        self.int_array = []
        self.vision_uncert_dict = None

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
        start_image_pos: Optional[List] = None,
        end_image_pos: Optional[List] = None,
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

            # ------------------begin: 计算图片开始与结束位置------------------#
            # start_image_pos是special_image_token_mask中为1的位置
            if start_image_pos is not None and end_image_pos is not None:
                start_image_pos.append(torch.nonzero(special_image_token_mask[-1] == 1).item())
                end_image_pos.append(start_image_pos[0] + num_image_features - 1)
            # ------------------end: 计算图片开始与结束位置------------------#

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
        if input_ids.shape[-1]!=1:
            self.is_first_generation = True
            self.logits_mask_prob = [] # restore logits mask prob
            self.start_image_pos = [] # restore start image pos
            self.end_image_pos = [] # restore end image pos
        else:
            self.is_first_generation = False
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
                # 获取start_image_pos 和 end_image_pos
                start_image_pos = []
                end_image_pos = []
                inputs_embeds, attention_mask, position_ids, labels, early_exit_output = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                    start_image_pos=self.start_image_pos,
                    end_image_pos=self.end_image_pos,
                )

                # print(f"dont use early stop exit on input embeddings")
                

            
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
        if self.is_first_generation:
            self.start_generation_pos = attention_mask.shape[-1] # 开始生成的位置
            self.do_step = True
            self.all_outputs = []

        
        def restore_attention_mask(attention_mask):
            attention_mask[:, :] = 1
            return attention_mask
        
        # kv会改变，需要深拷贝来进行储存
        original_past_key_values = copy.deepcopy(past_key_values)
        original_past_key_values_case = copy.deepcopy(past_key_values)
        output_hidden_states = True
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
        attention_mask = restore_attention_mask(attention_mask)
        outputs_all.append(outputs)
        logits = outputs['logits']
        if self.is_first_generation:
            self.int_array = []
            # 第一次生成，需要记录image feature
            self.image_features = self.get_image_features(self.start_image_pos, self.end_image_pos, outputs)
            vision_token_logits = self.get_image_logits(logits=logits, start_image_pos=self.start_image_pos[0],
                                                        end_image_pos=self.end_image_pos[0])
            self.vision_uncert_dict = self.calculate_vision_uncertainty(vision_token_logits)
            # write decoded image_features[1] to a file
            # with open("/home/fyx/hallucination/image_features_out.json", "w") as f:
            #     for i in range(self.image_features[1].shape[1]):
            #         # Decode the i-th token.
            #         token = self.processor.batch_decode(self.image_features[1][0][i].unsqueeze(0))
            #         # Get the corresponding variance value.
            #         variance_value = self.vision_uncert_dict["variance_per_token"][0][i]
            #         # Get other fields from vision_uncert_dict.
            #         epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0][i]
            #         alea_uncert = self.vision_uncert_dict["alea_uncert_per_token"][0][i]
            #
            #         # Write the data to the file.
            #         f.write(
            #             f"Token {i + 1}: {token}, Variance: {variance_value}, Epis_Uncert: {epis_uncert}, Alea_Uncert: {alea_uncert}\n")
            loss = None
        max_vote = True
        if not self.is_first_generation:
            outputs_all = [] # refresh outputs_all
            probs = [0.3,0.5,0.7]
            for prob in probs:
                original_past_key_values_ = copy.deepcopy(original_past_key_values)
                attention_mask = restore_attention_mask(attention_mask)
                attention_mask = self.get_image_attention_mask(logits, attention_mask, method="epis", prob=prob)
                outputs_all.append(
                    self.language_model(
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=original_past_key_values_,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                )
            loss = None
            # find the max logit in outputs_all and choose it
            # choose the sharpest logit distribution 
            if not max_vote:
                cur_maxlogit = -100
                for i in outputs_all:
                    if torch.max(i[0][-1]) > cur_maxlogit:
                        cur_maxlogit = torch.max(i[0][-1])
                        outputs_r = i
            elif None:
                cur_sharpness = float('inf')
                for i in outputs_all:
                    # 计算当前分布的标准差
                    std_dev = torch.std(i[0][-1])
                    # 比较分布的尖锐程度，选择最sharp的分布（标准差最小）
                    if std_dev < cur_sharpness:
                        cur_sharpness = std_dev
                        outputs_r = i
            if max_vote:
                outputs_r = select_by_vote(outputs_all)
            self.logits_mask_prob.append(1/torch.max(outputs_r[0][-1]).item())
            return LlavaNextCausalLMOutputWithPast(
                loss=loss,
                logits=outputs_r[0],
                past_key_values=outputs_r.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        loss = None
        self.logits_mask_prob.append(1/torch.max(logits[-1][-1]).item())
        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
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
        image_projected_features = logits[:, start_image_pos:end_image_pos+1]
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
        if len(start_image_pos)==0 or len(end_image_pos)==0:
            return None
        logits = outputs['logits']
        # print(start_image_pos, end_image_pos)
        image_logits = self.get_image_logits(logits=logits, start_image_pos=start_image_pos[0], end_image_pos=end_image_pos[0])
        values, ids = self.get_topk_token_id(image_logits, topk=10)
        image_features = (values, ids)
        return image_features
    
    def get_overlap_image_tokens(self,logits,id = -1):
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
    
    def get_image_attention_mask(self,logits,attention_mask,method="overlap", prob=0.5, ids=[]):
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
                return attention_mask
            attention_mask[:, adjusted_indices_tensor] = 1
        elif method == "VQA":
            for id in ids:
                matched_indices = self.get_overlap_image_tokens(logits,id)
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
            attention_mask[:, self.start_image_pos[0]:self.end_image_pos[0]+1] = 0
        elif method == "random_image":
            # mask based on random number
            for i in range(self.start_image_pos[0], self.end_image_pos[0]+1):
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
                if random_num  < self.logits_mask_prob[i]:
                    attention_mask[:, i + self.start_generation_pos] = 0
        elif method == "agressive":
            # agressively mask image token at a prob
            for i in range(self.start_image_pos[0], self.end_image_pos[0]+1):
                if i >= attention_mask.shape[-1]:
                    break
                random_num = torch.rand(1)
                if random_num < 0.2:
                    attention_mask[:, i] = 0
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

        return attention_mask
    
    def get_input(self, max_length):
        """
        Get input, input a list of int values between 0 and max_length, if the input is -1, then return the collected list.

        Args:
            max_length: max length

        Returns:
            array of int values
        """
        result = []

        while True:
            try:
                # 获取用户输入
                value = int(input(f"Enter an integer between 0 and {max_length} (or -1 to finish, -2 to skip all, -3 to input the skip step size, -4 for continuous mask, -5 for all image mask): "))
                
                # 如果输入为 -1，则返回当前收集到的 result 列表
                if value == -1:
                    return result
                if value == -2:
                    self.do_step = False
                    return []
                if value == -3:
                    self.skip_steps = int(input("Enter the step size: "))
                    return []
                if value == -4:
                    start = int(input("Enter the start position: "))
                    end = int(input("Enter the end position: "))
                    for i in range(start, end+1):
                        result.append(i)
                    return result
                if value == -5:
                    result.append(-1)
                    return result
                # 检查输入是否在有效范围内
                if 0 <= value <= max_length:
                    result.append(value)
                else:
                    print(f"Please enter a number between 0 and {max_length}.")
            
            except ValueError:
                print("Invalid input. Please enter an integer.")


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

