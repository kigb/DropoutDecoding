import copy
import math
from turtledemo.forest import start

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import Optional, Union, List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from models.config import settings
from models.llavanext import select_by_vote
from models.llava import select_by_average
# seed = 1141
seed = 5217
# seed = 650
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
# processor = InstructBlipProcessor.from_pretrained("/data4/fyx/instructblip-vicuna-7b")
start_img_pos = -1
end_img_pos = -1
start_generation_pos = -1
first_generation = False
pope_array = []
class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # self.processor = InstructBlipProcessor.from_pretrained("/data3/fyx/instructblip-vicuna-7b")
        self.start_image_pos = []
        self.end_image_pos = []
        self.start_generation_pos = -1
        self.token_entropies = []
        self.token_ventropies = []
        self.logits_mask_prob = []  # 用于text部分mask的概率
        self.image_features = None  # 用于记录图片特征
        self.first_generation = True
        self.vision_uncert_dict = None
        self.image_logits = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # voting_numbers: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(voting_numbers)
        global start_img_pos
        global end_img_pos
        global start_generation_pos
        global first_generation
        global pope_array
        if first_generation is True:
            self.first_generation = True
            first_generation = False
        original_past_key_values = copy.deepcopy(past_key_values)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.first_generation is True:
            self.start_image_pos = []
            self.end_image_pos = []
            self.start_image_pos.append(start_img_pos)
            self.end_image_pos.append(end_img_pos)
            self.image_features = self.get_image_features(self.start_image_pos, self.end_image_pos, outputs)
            self.start_generation_pos = start_generation_pos
            self.token_entropies = []
            self.token_ventropies = []
            self.logits_mask_prob = []
            vision_token_logits = self.get_image_logits(logits=logits, start_image_pos=self.start_image_pos[0],
                                                        end_image_pos=self.end_image_pos[0])
            self.image_logits = vision_token_logits
            self.vision_uncert_dict = self.calculate_vision_uncertainty(vision_token_logits)
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
        def restore_attention_mask(attention_mask):
            attention_mask[:, :] = 1
            return attention_mask

        if self.first_generation is not True:
            outputs_all = []
            mask_probs = settings['voting_numbers']
            text_mask_method = "logits"
            for mprob in mask_probs:
                original_past_key_values_ = copy.deepcopy(original_past_key_values)
                attention_mask = restore_attention_mask(attention_mask)
                attention_mask = self.get_image_attention_mask(logits, attention_mask, method="epis",prob=mprob)
                # attention_mask = self.get_image_attention_mask(logits, attention_mask, method="epis_kl",prob=mprob)
                # attention_mask = self.get_image_attention_mask(logits, attention_mask, method=text_mask_method)
                outputs_all.append(self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=original_past_key_values_,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position,
                ))
            if settings['use_avg'] is False:
                outputs_r = select_by_vote(outputs_all)
            else:
                outputs_r = select_by_average(outputs_all)
            hidden_states = outputs_r[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            self.logits_mask_prob.append(1 / torch.max(logits[0][-1]).item())
            en, ven = self.calculate_entropy_varentropy(logits[0][-1])
            self.token_entropies.append(en)
            self.token_ventropies.append(ven)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs_r.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        self.logits_mask_prob.append(1 / torch.max(logits[-1][-1]).item())
        en, ven = self.calculate_entropy_varentropy(logits[0][-1])
        self.token_entropies.append(en)
        self.token_ventropies.append(ven)
        self.first_generation = False
        return CausalLMOutputWithPast(
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
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        # print(start_image_pos, end_image_pos)
        image_logits = self.get_image_logits(logits=logits, start_image_pos=start_image_pos[0],
                                             end_image_pos=end_image_pos[0])
        values, ids = self.get_topk_token_id(image_logits, topk=10)
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

    def calculate_entropy_varentropy(self, logits) -> (float, float):
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
            max_ids = torch.tensor([[id]], device=self.image_features[1].device)
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
                    attention_mask[:, i + self.start_generation_pos] = (random_num>0.5)
                else :
                    random_num = torch.rand(1)
                    if random_num < self.logits_mask_prob[i]:
                        attention_mask[:, i + self.start_generation_pos] = 0
            attention_mask[:, -3:] = 1
        elif method == "agressive":
            # agressively mask image token at a prob
            for i in range(self.start_image_pos[0], self.end_image_pos[0] + 1):
                if i >= attention_mask.shape[-1]:
                    break
                random_num = torch.rand(1)
                if random_num < prob:
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
            # epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]
            #
            # # 获取 epis_uncert 的分位数
            # q_low = torch.quantile(epis_uncert, 0)
            # q_high = torch.quantile(epis_uncert, 1)
            #
            #
            # # clamp 是为了确保数据在分位数范围内，并进行插值
            # normalized_probs = 0.1 + (prob-0.1) * (epis_uncert.clamp(min=q_low, max=q_high) - q_low) / (q_high - q_low)
            #
            # # 生成与 epis_uncert 相同形状的随机张量
            # random_tensor = torch.rand_like(epis_uncert)
            #
            # # 根据生成的随机张量与 normalized_probs 确定 mask
            # mask = random_tensor < normalized_probs
            # filtered_indexes = torch.where(mask)[0]
            #
            # # 设置 attention_mask 中的指定位置为 0
            # attention_mask[:, filtered_indexes + self.start_image_pos[0]] = 0
            epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]

            # Step 1: Determine the threshold value for the top `prob` proportion
            threshold = torch.quantile(epis_uncert, 1 - prob)

            # Step 2: Create a mask that identifies tokens with epis_uncert values in the top `prob` proportion
            mask = epis_uncert >= threshold
            filtered_indexes = torch.where(mask)[0]

            # Step 3: Set specified positions in attention_mask to 0 based on the mask
            attention_mask[:, filtered_indexes + self.start_image_pos[0]] = 0

            # Step 4: Restore specific indices in attention_mask to 1, if needed
            attention_mask[:, adjusted_indices_tensor] = 1

            # # 恢复 attention_mask 的某些索引为 1
            # attention_mask[:, adjusted_indices_tensor] = 1
        elif method=="epis_kl":
            epis_uncert = self.vision_uncert_dict["epis_uncert_per_token"][0]

            # 获取 epis_uncert 的分位数
            q_low = torch.quantile(epis_uncert, 0)
            q_high = torch.quantile(epis_uncert, 1)

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
            lowest_kls = self.lowest_percent_kl_indices(self.image_logits,logits)
            lowest_kls_indices = (lowest_kls + self.start_image_pos[0]).tolist()  # 转换为 Python 列表
            attention_mask[:,lowest_kls_indices] = 1
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
        return attention_mask

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

class CustomInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            self.language_model = CustomLlamaForCausalLM._from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )
            # self.processor = InstructBlipProcessor.from_pretrained("/data3/fyx/instructblip-vicuna-7b")

        @torch.no_grad()
        def generate(
                self,
                pixel_values: torch.FloatTensor,
                qformer_input_ids: Optional[torch.LongTensor] = None,
                qformer_attention_mask: Optional[torch.LongTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                interpolate_pos_encoding: bool = False,
                **generate_kwargs,
        ) -> torch.LongTensor:
                global start_img_pos
                start_img_pos = 0
                global first_generation
                first_generation = True
                if hasattr(self, "hf_device_map"):
                        # preprocess for `accelerate`
                        self._preprocess_accelerate()

                batch_size = pixel_values.shape[0]
                image_embeds = self.vision_model(
                        pixel_values,
                        return_dict=True,
                        interpolate_pos_encoding=interpolate_pos_encoding,
                ).last_hidden_state

                image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                                                  device=image_embeds.device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long,
                                                  device=image_embeds.device)
                if qformer_attention_mask is None:
                        qformer_attention_mask = torch.ones_like(qformer_input_ids)
                qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
                query_outputs = self.qformer(
                        input_ids=qformer_input_ids,
                        attention_mask=qformer_attention_mask,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_attention_mask,
                        return_dict=True,
                )
                query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

                language_model_inputs = self.language_projection(query_output)
                language_attention_mask = torch.ones(
                        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
                )

                if input_ids is None:
                        input_ids = (
                                torch.LongTensor([[self.config.text_config.bos_token_id]])
                                .repeat(batch_size, 1)
                                .to(image_embeds.device)
                        )
                if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)],
                                           dim=1)
                global end_img_pos
                end_img_pos = language_attention_mask.shape[-1] - 1
                global start_generation_pos
                start_generation_pos = attention_mask.shape[-1]
                # pope
                data = input_ids[0]
                global pope_array
                indices_32000 = (data == 727).nonzero(as_tuple=True)[0]
                if indices_32000.numel() > 0:
                    index_32000 = indices_32000[0].item()
                    elements_after_32000 = data[index_32000 + 1:]
                    pope_array = elements_after_32000.cpu().numpy()
                # concatenate query embeddings with prompt embeddings
                inputs_embeds = self.get_input_embeddings()(input_ids)
                # print(inputs_embeds.shape)
                inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
                                          dim=1)
                # print(inputs_embeds.shape)
                # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
                # -1 is to account for the prepended BOS after `generate.`
                if not self.language_model.config.is_encoder_decoder:
                        generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + \
                                                        language_model_inputs.shape[1] - 1
                        generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + \
                                                        language_model_inputs.shape[1]
                # print(self.language_model.config)
                outputs = self.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        **generate_kwargs,
                )

                # this is a temporary workaround to be consistent with other generation models and
                # have BOS as the first token, even though under the hood we are calling LM with embeds
                if not self.language_model.config.is_encoder_decoder:
                        # the InstructBLIP authors used inconsistent tokenizer/model files during training,
                        # with the tokenizer's bos token being set to </s> which has ID=2,
                        # whereas the model's text config has bos token id = 0
                        bos_token_id = (
                                2
                                if self.config.text_config.architectures[0] == "LLaMAForCausalLM"
                                else self.config.text_config.bos_token_id
                        )
                        bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
                        if not isinstance(outputs, torch.Tensor):
                                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
                        else:
                                outputs = torch.cat([bos_tokens, outputs], dim=-1)

                return outputs
