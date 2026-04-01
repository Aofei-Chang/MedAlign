#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# import sys
# sys.path.insert(0, '/opt/conda/lib/python3.10')
# print(sys.path)
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         Qwen2ForCausalLM, Qwen2Config, Qwen2Model \

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, init_vision_encoder_from_ckpt=False, tuple_params=None):
        # config._attn_implementation = "flash_attention_2"
        # config._flash_attn_2_enabled = True
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        # assert self.model._use_flash_attention_2 == True
        # self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path,padding_side="left")
        self.tokenizer = None
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.tuple_params = tuple_params

        # if getattr(config, 'init_vision_encoder_from_ckpt', True):
        if init_vision_encoder_from_ckpt:
            vision_tower = self.get_vision_tower()
            print(f'loading from CLIP first. This should only be used at inference!!!')
            vision_tower.load_model() # 
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        spatial_cos_matrix = None,
        attention_map_label = None,
        spatial_loss_weight = None,
        image_token_start_index = None,
        question_token_end_index = None,
        # VCD_parameters
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        # avisc parameters
        img_idx: Optional[Tuple] = None,
        mask_idx: Optional[torch.Tensor] = None,
        use_avisc: Optional[bool] = None,
        layer_gamma=None,
        masking_scheme=None,
        lamb=None,
        temp=None,
        use_m3id=None,
        use_damro=False,
        out_vit_attention=False,
        # DoLa parameters
        early_exit_layers: Optional[List[int]] = None,
        # **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print(f"input_ids: {input_ids.shape}, {input_ids}, {labels}")
        damro_mask_idx = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                damro_mask_idx
            # ) = self.prepare_inputs_labels_for_multimodal(
            ) = self.prepare_inputs_labels_for_multimodal_new(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                out_vit_attention=out_vit_attention
            )
        self.damro_mask_idx = damro_mask_idx
        # print(f"attention_mask: {attention_mask.shape}, inputs_embeds: {inputs_embeds.shape}, labels: {labels.shape}, {labels}")
        if self.damro_mask_idx is not None and past_key_values is None:
            mask_idx = self.damro_mask_idx
        if image_token_start_index is None:
            image_token_start_index = 34
        if mask_idx is not None and past_key_values is None and masking_scheme is not None:
            # top-k masking
            # for att_mask, idx in zip(attention_mask, mask_idx):
            #     att_mask[idx] = 0
            # print(f"use damro masks", mask_idx.size())
            #token noising    
            for input_embed, idx in zip(inputs_embeds, mask_idx):
                # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
                #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
                if masking_scheme.lower() == "ones":
                    input_embed[idx + image_token_start_index] = 1.0
                    # print("ones")
                elif masking_scheme.lower() == "zeros":
                    input_embed[idx + image_token_start_index] = 0.0
                    # print("zeros")
                elif masking_scheme.lower() == "noise":
                    input_embed[idx + image_token_start_index] = torch.randn(input_embed[idx + 35].size(), dtype=input_embed.dtype).to(input_embed.device)
                    # print("noise")
                else:
                    input_embed[idx + image_token_start_index] = 0.0
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            spatial_cos_matrix=spatial_cos_matrix, 
            attention_map_label=attention_map_label,
            spatial_loss_weight=spatial_loss_weight,
            tuple_params=self.tuple_params,
            image_token_start_index = image_token_start_index,
            question_token_end_index = question_token_end_index,
            early_exit_layers=early_exit_layers,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) :
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        use_damro = False
        if "use_damro" in kwargs:
            use_damro = kwargs.get("use_damro")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                damro_mask_idx
            ) = self.prepare_inputs_labels_for_multimodal_new(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                out_vit_attention=use_damro,
            )
            self.damro_mask_idx = damro_mask_idx   
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # print(inputs_embeds.shape)
        return super().generate(
            position_ids=None,
            attention_mask=None,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
    def prepare_inputs_for_generation_method(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    #for baseline VCD
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        prepared_inputs_ = kwargs.get("images_cd", None)

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs

    def prepare_inputs_for_generation_m3id(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            
            # print(inputs_embeds.size(), "inputs_embeds")
            # here is a bug, the inputs_embeds should not contain the original image input
            # model_inputs = {"inputs_embeds": inputs_embeds}
            inputs_embeds = torch.cat([inputs_embeds[:,:34,:], inputs_embeds[:,34+576:,:]], dim=1)
            # print(inputs_embeds.size(), "inputs_embeds")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids[input_ids != -200].unsqueeze(0)}
        # print(model_inputs, "nbij")
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask[:,:-1],
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
# AutoConfig.register("llava", LlavaQwenConfig)
AutoConfig.register("llava_qwen2", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwen2ForCausalLM)