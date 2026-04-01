import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append("/home/xxx/research/MedAlign/huatuoGPT_V")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_images

from llava.model_wrapper import LayerWrapper, AttnWrapper
from llava.moe_llava import LoRA_MOE_Q

from PIL import Image
import math
from transformers import set_seed, logging, AutoTokenizer

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_path)

    if args.baseline == "VCD":
        from VCD_files.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()
    elif args.baseline == "avisc" or args.baseline == "m3id" or args.baseline == "damro":
        print(f"use and set {args.baseline} sampling")
        from avisc_utils.vcd_add_noise import add_diffusion_noise
        from avisc_utils.avisc_sample import evolve_avisc_sampling
        evolve_avisc_sampling()

    if '7b' in model_name.lower():
        print(f'loading from {model_name}')
        from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
        model, loading_info = LlavaQwen2ForCausalLM.from_pretrained(model_name, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
        model = model.to("cuda")
        missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
        unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
        assert all(['vision_tower' in k for k in unexpected_keys])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
        # self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(model_name)
        vision_tower.to(dtype=torch.bfloat16, device=model.device)
        image_processor = vision_tower.image_processor
        
    elif 'huatuogpt' in model_name.lower():
        print(f'loading from {model_name}')
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        model, loading_info = LlavaLlamaForCausalLM.from_pretrained(model_name, init_vision_encoder_from_ckpt=True, output_loading_info=True, torch_dtype=torch.bfloat16)
        missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
        unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
        assert all(['vision_tower' in k for k in unexpected_keys])

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
        # self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(model_name)
        vision_tower.to(dtype=torch.bfloat16, device=model.device)
        image_processor = vision_tower.image_processor

    else:
        raise NotImplementedError
    

    #LOAD LORA
    peft_path = args.peft_path
    if peft_path is not None and len(str(peft_path))>4:
        from peft import PeftModel
        print(f"Loading LoRA weights from {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        print(f"Merging weights")
        model = model.merge_and_unload()
        
        steer_path = os.path.join(peft_path, "non_lora_trainables.bin")
        steer_state_dict = torch.load(steer_path, map_location='cuda')
        if len(steer_state_dict.keys()) >= 1:
            # set steer modules
            align_layers = [int(i) for i in args.steer_w_layer.split(',')] if args.steer_w_layer is not None else []
            SAE_layers = [int(i) for i in args.SAE_layer.split(',')] if (args.SAE_layer is not None and args.SAE_layer != "None") else []
            # print(model)
            for i in SAE_layers:
                if i > 0 and args.moe_num_experts>1:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_Q(args=None,
                            lora_rank=16,
                            lora_alpha=8,
                            num_experts=args.moe_num_experts,
                            original_module=original_q,
                            dense_moe=True)

            for layer in SAE_layers:
                use_adapters=False
                if layer > 0:
                    print(f"Attn steer weights are applied to layer {layer}")
                    # model.model.layers[layer].self_attn = QueryGatedAdapter(model.model.layers[layer].self_attn, hidden_dim=4096, bottleneck_dim=args.bottleneck_dim, num_adapters=args.num_adapters, use_adapters=use_adapters)
                    model.model.layers[layer].self_attn = AttnWrapper(model.model.layers[layer].self_attn)
            hidden_dim = model.config.hidden_size
            for layer in align_layers:
                use_adapters=True
                if layer > 0:
                    print(f"Steer weights are applied to layer {layer}")
                    W = torch.zeros((hidden_dim, hidden_dim), device=model.device, dtype=model.dtype)
                    W = torch.nn.Parameter(W)
                    model.model.layers[layer - 1] = LayerWrapper(model.model.layers[layer - 1], W=W, epsilon=1.0, use_adapters=use_adapters)

            new_state_dict = {}
            for key, value in steer_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                print(key)
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
            model.load_state_dict(new_state_dict, strict=False)
            print("Steering modules loaded successfully.")
            model = model.to("cuda")
            model.to(torch.bfloat16)

    if args.baseline == "PAI":
        from transformers.generation.logits_process import LogitsProcessorList
        from PAI_files.model_loader import init_cfg_processor
        from PAI_files.attention import llama_modify
        llama_modify(
            model,
            start_layer=2,
            end_layer=28,
            alpha=0.2,
            use_attn=True,
            use_cfg=True,
            img_start_idx=34,
            img_end_idx=34+576
        )

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        if "conversations" in line:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1]['value'] # ['value']        
            qs = question['value']
        else:
            question = line['question']
            gt_ans = line['answer']
            qs = question
            if 'choices' in line and len(line['choices']) > 10:
                qs += " Please choose from the following options: " + line['choices']
            if 'question_type' in line and line['question_type'] == 'binary':
                qs += " Please answer Yes or No."

        if "qid" in line:
            idx = line["qid"]
        else:
            idx = line["id"]
        # question = line["conversations"][0] # ['value'].split('\n')[0]
        # gt_ans = line["conversations"][1] # ['value']      
        # image_file = line["image"]
        # assert gt_ans['from'] == 'gpt'
        if 'image' in line:
            image_file = line["image"]
        elif 'img_name' in line:
            image_file = line["img_name"]

        # qs = question['value']
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # print(args.conv_mode, "model mode")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(f"Prompt: {prompt}")
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep2
        # keywords = [stop_str, "\n", "\n\n", "."]
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        if args.baseline == "opera":
            key_position = {
                "image_start": 34, 
                "image_end": 34+576,
                "response_start": input_ids.shape[1] + 576 - 2
            }
            # print("use opera")
            # print(tokenizer.eos_token_id, tokenizer.pad_token_id)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                    num_beams=5,
                    do_sample=False,
                    max_new_tokens=128,
                    output_attentions=True,
                    opera_decoding=True,
                    key_position=key_position,
                    scale_factor=50,
                    threshold=15,
                    num_attn_candidates=5,
                    penalty_weights=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "PAI":
            logits_processor = (
                    init_cfg_processor(tokenizer=tokenizer, llm_model=model, questions=[cur_prompt], gamma=1.1, beam=1, start_layer=2, end_layer=28)
                )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1024,
                    output_attentions=False,
                    output_hidden_states=False,
                    logits_processor=LogitsProcessorList([logits_processor]),
                    stopping_criteria=[stopping_criteria])     
        
        elif args.baseline == "VCD":
            # print("use VCD")
            from VCD_files.vcd_add_noise import add_diffusion_noise
            image_tensor_cd = add_diffusion_noise(image_tensor, 500) #args.noise_step
            # image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                        images_cd=(image_tensor_cd.unsqueeze(0).to(torch.bfloat16).cuda() if image_tensor_cd is not None else None),
                        cd_alpha = 1, #args.cd_alpha,
                        cd_beta = 0.1, #args.cd_beta,
                        do_sample=True,
                        max_new_tokens=1024,
                        temperature=1,
                        output_attentions=False,
                        output_hidden_states=False,
                        stopping_criteria=[stopping_criteria]
                        )
        elif args.baseline == "avisc":
            use_cd = False
            if use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
            else:
                image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                        images_cd=(image_tensor_cd.to(torch.bfloat16).cuda() if image_tensor_cd is not None else None),
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_avisc=True,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "m3id":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                        images_cd=None,
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_avisc=False,
                        use_m3id=True,
                        layer_gamma=0.5,
                        lamb=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "DoLa":
            with torch.no_grad():
                early_exit_layers = [0,2,4,6,8,10,12,14,28]
                mature_layer = early_exit_layers[-1]
                premature_layer = None
                candidate_premature_layers = early_exit_layers[:-1]
                output_ids = model.generate(input_ids, 
                                            images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                                            max_new_tokens=1024,
                                            dola_decoding=True,
                                            top_p=0.95, top_k=0, temperature=0.9, 
                                            stopping_criteria=[stopping_criteria], relative_top=0.1, 
                                            mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers,
                                            )
        elif args.baseline == "damro":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                        images_cd=None,
                        cd_alpha=0.5,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_damro=True,
                        use_avisc=False,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "beam":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                    do_sample=False,
                    num_beams=5,
                    max_new_tokens=128,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "greedy":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=128,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "nucleus":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(),
                    do_sample=True,
                    top_p=0.9,                 # Cumulative probability threshold
                    top_k=0,                   # Optional; setting to 0 disables top-k filtering
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        else:

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.cuda(),
                    images=image_tensor.unsqueeze(0).cuda().to(torch.bfloat16),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    # do_sample=False,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # num_beams=5,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=256,
                    stopping_criteria=[stopping_criteria],
                    use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "gt_ans": gt_ans,
                                #    "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--peft-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--steer_w_path", type=str, default=None)
    parser.add_argument("--steer_w_layer", type=str, default="22")
    parser.add_argument("--SAE_layer", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)