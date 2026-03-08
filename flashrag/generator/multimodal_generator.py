from typing import List, Dict, Union
import os
import json
import importlib
from copy import deepcopy
import warnings
import math
import torch.nn.functional as F
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModel
from abc import abstractmethod
import json
import os
import importlib
import base64
from io import BytesIO
from flashrag.generator.utils import convert_image_to_base64, process_image, resolve_max_tokens, process_image_pil
import time
class BaseMultiModalGenerator:
    """`BaseMultiModalGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.model_path = config["generator_model_path"]

        self.max_input_len = config["generator_max_input_len"]
        self.batch_size = config["generator_batch_size"]
        self.device = config["device"]
        self.gpu_num = torch.cuda.device_count()
        self.config = config
        self.generation_params = config["generation_params"]
    
    def generate(self, input_list: list):
        """
        input_list: A list contains of messages, each message is a list, like:
        [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "image1_path"},
                    {"type": "image", "image": "image2_path"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        
        The content of each dict can be a str(pure text as input) or list(for multimodal input).
        """
        
        pass

class BaseInferenceEngine:
    def __init__(self, model_path, device='cpu', max_input_len=4096):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.max_input_len = max_input_len
        self._load_model()

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    # @torch.inference_mode(mode=True)
    def generate(self, input_list: list, batch_size=None, **params):
        pass

class Qwen2_5VLInferenceEngine(BaseInferenceEngine):
    from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    def _load_model(self):
        self.model = self.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            # device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._cost_stats = []

    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        # convert image to base64
        for messages in input_list:
            for message in messages:
                if isinstance(message['content'], list):
                    for content_dict in message['content']:
                        if content_dict['type'] == 'image':
                            content_dict['image'] = convert_image_to_base64(content_dict['image'])
        from qwen_vl_utils import process_vision_info
        texts = [self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in input_list]
        image_inputs, video_inputs = process_vision_info(input_list)
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        # input token统计
        input_tokens = [len(input_id) for input_id in inputs.input_ids]
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            **params
        )
        end_time = time.time()
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        output_tokens = [len(out_ids) for out_ids in generated_ids_trimmed]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # 记录token消耗
        for inp, outp in zip(input_tokens, output_tokens):
            self._cost_stats.append({"input_tokens": inp, "output_tokens": outp, "total_tokens": inp+outp, "latency(s)": end_time - start_time})
        return output_text
    def cost_stats(self):
        cost_stats = self._cost_stats
        self._cost_stats = []
        return cost_stats
    
class Qwen2VLInferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        from transformers import Qwen2VLForConditionalGeneration
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True
        ).eval()
        min_pixels = 3136
        max_pixels = 12845056
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels)
        self.processor.tokenizer.model_max_length = self.max_input_len
        self.tokenizer = self.processor.tokenizer
    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        # convert image to base64
        for messages in input_list:
            for message in messages:
                if isinstance(message['content'], list):
                    for content_dict in message['content']:
                        if content_dict['type'] == 'image':
                            content_dict['image'] = convert_image_to_base64(content_dict['image'])

        from qwen_vl_utils import process_vision_info
        texts = [self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in input_list]
        image_inputs, video_inputs = process_vision_info(input_list)    
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        # print(inputs)
        # print(inputs['input_ids'].shape,inputs['attention_mask'].shape,inputs['pixel_values'].shape,inputs['image_grid_thw'].shape)
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **params
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
    
class InternVL2InferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        import torch
        gpu_num = torch.cuda.device_count()
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto' if gpu_num <= 1 else self.split_model(),
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.model_max_length = self.max_input_len
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id 


    def split_model(self):
        # get model name
        with open(os.path.join(self.model_path, 'config.json')) as f:
            config = json.load(f)
        num_layers = config['llm_config']['num_hidden_layers']
        device_map = {}
        world_size = torch.cuda.device_count()
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def build_transform(self, input_size):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        import torch
        # TODO: Currently only support single image batch or multi-image without batch 
        # convert input format to internvl2
        final_prompt_list = [] # each is a str
        final_image_list = [] # each is a list
        system_message = None
        for messages in input_list:
            # parse each query
            for item in messages:
                if item['role'] == 'system':
                    assert isinstance(item['content'], str)
                    system_message = item['content']
                else:
                    if isinstance(item['content'], str):
                        # pure text input
                        final_prompt_list.append(item['content'])
                        final_image_list.append([])
                    else:
                        # multimodal input
                        image_list = [d['image'] for d in item['content'] if d['type'] == 'image']
                        text = [d['text'] for d in item['content'] if d['type'] == 'text'][0]
                        final_prompt_list.append(text)
                        final_image_list.append(image_list)
        if system_message is not None:
            self.model.system_message = system_message
        
        final_image_list = [[self.load_image(img, max_num=12).to(self.model.dtype).to(self.model.device) for img in image_list] for image_list in final_image_list]

        if all([len(image_list) ==1 for image_list in final_image_list]):
            torch.cuda.empty_cache()
            # batch inference with single image
            final_image_list = [image_list[0] for image_list in final_image_list]
            pixel_values = torch.cat(final_image_list, dim=0)
            num_patches_list = [img.size(0) for img in final_image_list]
            final_prompt_list = [f'<image>\n{text}' for text in final_prompt_list]
            outputs = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                final_prompt_list,
                num_patches_list=num_patches_list,
                generation_config=params,
                history=None,
                return_history=None,
            )
            return outputs
        else:
            # do single item inference with multi image 
            outputs = []
            for image_list, prompt in zip(final_image_list, final_prompt_list):
                torch.cuda.empty_cache()
                pixel_values = torch.cat(image_list, dim=0)
                num_patches_list = [img.size(0) for img in image_list]
                prompt_prefix = ""
                for i in range(len(image_list)):
                    prompt_prefix += f'Image-{i+1}: <image>\n'
                prompt = prompt_prefix + prompt

                output = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    num_patches_list=num_patches_list,
                    generation_config=params,
                    history=None,
                    return_history=None,
                )
                outputs.append(output)
            return outputs

class LlavaInferenceEngine(BaseInferenceEngine):
    def _load_model(self):
        from transformers import AutoProcessor
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            config = json.load(f)
            model_type = config['architectures'][0]
        model_type = getattr(importlib.import_module('transformers'), model_type)
        self.model = model_type.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        self.tokenizer = self.processor.tokenizer
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.image_token = "<image>"

    @torch.inference_mode(mode=True)
    def generate(self, input_list, **params):
        # add special tokens
        new_input_list = []
        visual_list = []
        for messages in input_list:
            new_messages = []
            for message in messages:
                item_visual_list = []
                if isinstance(message['content'],list):
                    # remove all image
                    image_list = [item['image'] for item in message['content'] if item['type'] == 'image']
                    item_visual_list.extend(image_list)
                    text_content = [item['text'] for item in message['content'] if item['type'] == 'text'][0]
                    image_tokens = " ".join([self.image_token]*len(image_list))
                    text_content = f"{image_tokens}\n{text_content}"
                    new_messages.append({"role": message['role'], "content": text_content})
                else:
                    new_messages.append(message)
                visual_list.append(item_visual_list)
            new_input_list.append(new_messages)
        visual_list = sum(visual_list, [])
        texts = self.tokenizer.apply_chat_template(new_input_list, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=texts, images=visual_list, padding=True, truncation=True, max_length=self.max_input_len, return_tensors='pt').to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **params
        )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        output_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return output_text


class HFModelInferenceEngineFactory:
    _engine_map = {
        'qwen2_5_vl': Qwen2_5VLInferenceEngine,
        'qwen': Qwen2VLInferenceEngine,
        'llava': LlavaInferenceEngine,
        'internvl': InternVL2InferenceEngine,
    }

    @staticmethod
    def get_engine(model_path, device='cuda', **kwargs):
        config_file_path = os.path.join(model_path, 'config.json')
        with open(config_file_path, "r") as f:
            model_config = json.load(f)
        model_arch = model_config['architectures'][0]
    
        for engine_name, engine_class in HFModelInferenceEngineFactory._engine_map.items():
            if engine_name in model_arch.lower():
                return engine_class(model_path, device, **kwargs)
            
        raise ValueError(f"Model {model_path} is not supported!")
      
    
class HFMultiModalGenerator(BaseMultiModalGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_name = config['generator_model']
        self.model_path = config['generator_model_path']
        self.inference_engine = HFModelInferenceEngineFactory.get_engine(
            model_path=self.model_path,
            device=self.device,
            max_input_len=self.max_input_len
        )

    @torch.inference_mode(mode=True)
    def generate(
        self,
        input_list: list,
        batch_size=None,
        **params
    ):
        # solve params
        if not isinstance(input_list[0], list):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        should_return_dict = generation_params.pop('return_dict', False)
        if 'temperature' not in generation_params:
            generation_params['temperature'] = 0
        if 'do_sample' not in generation_params:
            generation_params['do_sample'] = True if generation_params['temperature'] > 0 else False
        if generation_params['do_sample'] == False:
            generation_params['temperature'] = 0
        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from flashrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # preprocess input list
        from PIL import Image
        for messages in input_list:
            for message in messages:
                if isinstance(message['content'], list):
                    for content_dict in message['content']:
                        if content_dict['type'] == 'image':
                            content_dict['image'] = process_image_pil(content_dict['image'])

        output_responses = []
        for idx in trange(0, len(input_list), batch_size, desc='Generation process: '):
            torch.cuda.empty_cache()
            batch_prompts = input_list[idx: idx+batch_size]
            output_responses.append(self.inference_engine.generate(batch_prompts, **generation_params))
        if should_return_dict:
            return {"responses": output_responses}
        else:
            return output_responses

class VLLMMMGenerator(BaseMultiModalGenerator):
    """Class for decoder-only multimodal generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        self.use_lora = False
        self._perf_stats = []
        # vLLM related setting
        self.tensor_parallel_size = config['tensor_parallel_size'] if 'tensor_parallel_size' in config else 1
        self.gpu_memory_utilization = config['vllm_gpu_memory_utilization'] if 'vllm_gpu_memory_utilization' in config else 0.85
        self.max_model_len = config['max_model_len'] if 'max_model_len' in config else self.max_input_len
        self.limit_mm_per_prompt = config['limit_mm_per_prompt'] if 'limit_mm_per_prompt' in config else {"image": 5}
        self.mm_processor_kwargs = config['mm_processor_kwargs'] if 'mm_processor_kwargs' in config else {"min_pixels": 3136, "max_pixels": 12845056}
        self.enforce_eager = config['vllm_enforce_eager'] if 'vllm_enforce_eager' in config else True

        try:
            from transformers.configuration_utils import PretrainedConfig
            if not hasattr(PretrainedConfig, "standardize_rope_params"):
                PretrainedConfig.standardize_rope_params = lambda self: None
            if not hasattr(PretrainedConfig, "validate_rope"):
                PretrainedConfig.validate_rope = lambda self: None
        except Exception:
            pass

        from vllm import LLM
        mm_config = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "trust_remote_code": True,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
            "mm_processor_kwargs": self.mm_processor_kwargs,
            "enforce_eager": self.enforce_eager,
        }

        if self.use_lora:
            mm_config.update({
                "enable_lora": True,
                "max_lora_rank": 64,
                "max_logprobs": 32016,
            })

        self.model = LLM(**mm_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _record_perf(self, outputs, batch_size, latency_seconds):
        input_tokens = 0
        output_tokens = 0

        for output in outputs:
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            if prompt_token_ids is not None:
                input_tokens += len(prompt_token_ids)

            output_items = getattr(output, "outputs", [])
            if len(output_items) > 0:
                token_ids = getattr(output_items[0], "token_ids", None)
                if token_ids is not None:
                    output_tokens += len(token_ids)

        total_tokens = input_tokens + output_tokens
        stat = {
            "batch_size": batch_size,
            "latency_seconds": latency_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_per_second": (input_tokens / latency_seconds) if latency_seconds > 0 else 0.0,
            "output_tokens_per_second": (output_tokens / latency_seconds) if latency_seconds > 0 else 0.0,
            "total_tokens_per_second": (total_tokens / latency_seconds) if latency_seconds > 0 else 0.0,
            "timestamp": time.time(),
        }
        self._perf_stats.append(stat)

    def get_performance_stats(self, reset=False):
        stats = self._perf_stats
        run_count = len(stats)
        total_samples = sum(item["batch_size"] for item in stats)
        total_latency = sum(item["latency_seconds"] for item in stats)
        total_input_tokens = sum(item["input_tokens"] for item in stats)
        total_output_tokens = sum(item["output_tokens"] for item in stats)
        total_tokens = sum(item["total_tokens"] for item in stats)

        result = {
            "runs": run_count,
            "total_samples": total_samples,
            "total_latency_seconds": total_latency,
            "avg_latency_seconds_per_run": (total_latency / run_count) if run_count > 0 else 0.0,
            "avg_latency_seconds_per_sample": (total_latency / total_samples) if total_samples > 0 else 0.0,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_per_second": (total_input_tokens / total_latency) if total_latency > 0 else 0.0,
            "output_tokens_per_second": (total_output_tokens / total_latency) if total_latency > 0 else 0.0,
            "total_tokens_per_second": (total_tokens / total_latency) if total_latency > 0 else 0.0,
            "last_run": stats[-1] if run_count > 0 else None,
            "history": stats,
        }

        if reset:
            self._perf_stats = []

        return result

    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config['generator_max_input_len']
    from vllm.multimodal import MultiModalDataDict
    def _prepare_multimodal_input(self, input_data: Union[str, List[Dict]]) -> Dict:
        """
        解析标准 Qwen messages 格式
        input_data 示例: 
        [
            {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "Describe this image."}]}
        ]
        """
        if isinstance(input_data, str):
            return {"prompt": input_data}

        if isinstance(input_data, list):
            images = []
            for message in input_data:
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            # 支持 "image" 键或 OpenAI 风格的 "image_url" 键
                            img_source = item.get("image") or item.get("image_url", {}).get("url")
                            if img_source:
                                images.append(img_source)
                elif isinstance(content, str):
                    continue

            # 2. 使用 Tokenizer 应用对话模板
            # apply_chat_template 会自动处理 <|im_start|>, <|vision_start|> 等特殊标记
            prompt = self.tokenizer.apply_chat_template(
                input_data,
                tokenize=False,
                add_generation_prompt=True
            )

            # 3. 构造 vLLM 多模态输入字典
            mm_data = {}
            if images:
                # 如果是多图，传入列表；单图则传入单个元素
                mm_data["image"] = images[0] if len(images) == 1 else images

            return {
                "prompt": prompt,
                "multi_modal_data": mm_data
            }
        
        return {"prompt": str(input_data)}
    def generate(
        self,
        input_list: List[Union[str, Dict]], # 支持字符串或包含图像信息的字典
        return_raw_output=False,
        return_scores=False,
        **params,
    ):
        from vllm import SamplingParams

        if isinstance(input_list, (str, dict)):
            input_list = [input_list]

        # 1. 构建符合 vLLM 多模态要求的输入列表
        vllm_inputs = [self._prepare_multimodal_input(item) for item in input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        
        if not generation_params.get("do_sample", True):
            generation_params["temperature"] = 0
        generation_params.pop("do_sample", None)
        
        generation_params["seed"] = self.config['seed'] if 'seed' in self.config else 42

        # 处理停止词 (针对 Qwen/Llama 系列)
        stop_words = generation_params.get("stop", [])
        for sw in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            if sw not in stop_words:
                stop_words.append(sw)
        generation_params["stop"] = stop_words

        if return_scores:
            generation_params["logprobs"] = generation_params.get("logprobs", 5)

        sampling_params = SamplingParams(**generation_params)
        start_time = time.time()

        # 2. 调用 vLLM 推理
        if self.use_lora:
            from vllm.lora.request import LoRARequest
            outputs = self.model.generate(
                vllm_inputs,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(vllm_inputs, sampling_params)
        end_time = time.time()
        self._record_perf(outputs, len(vllm_inputs), end_time - start_time)

        # 3. 后处理输出
        if return_raw_output:
            return outputs

        generated_texts = [output.outputs[0].text for output in outputs]
        
        if return_scores:
            scores = []
            for output in outputs:
                try:
                    generated = output.outputs[0]
                    token_ids = getattr(generated, "token_ids", [])
                    token_logprobs = getattr(generated, "logprobs", [])

                    prob_list = []
                    for token_id, logprob_dict in zip(token_ids, token_logprobs):
                        selected = None
                        if isinstance(logprob_dict, dict):
                            selected = logprob_dict.get(token_id)
                            if selected is None and len(logprob_dict) > 0:
                                selected = max(logprob_dict.values(), key=lambda x: getattr(x, "logprob", float("-inf")))
                        if selected is None:
                            continue
                        prob_list.append(float(np.exp(selected.logprob)))

                    scores.append(prob_list)
                except:
                    scores.append([])
            return generated_texts, scores
        
        return generated_texts







