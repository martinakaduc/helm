from threading import Lock
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from dataclasses import dataclass
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedGraniteSpeechProcessor:
    """Loaded model and processor for Granite Speech."""

    model: AutoModelForSpeechSeq2Seq
    tokenizer: AutoProcessor
    generation_config: GenerationConfig


_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedGraniteSpeechProcessor]] = {
    "ibm-granite/granite-speech-3.3-8b": None,
    "ibm-granite/granite-speech-3.3-2b": None,
    "ibm-granite/granite-speech-3.2-8b": None,
}


class GraniteSpeechClient(CachingClient):
    """
    From https://huggingface.co/ibm-granite/granite-speech-3.3-8b,
    https://huggingface.co/ibm-granite/granite-speech-3.3-2b,
    https://huggingface.co/ibm-granite/granite-speech-3.2-8b,
    Granite Speech is a large multimodal language model (LLM) that can process and generate text, audio, and images.
    It is designed to be a general-purpose model for a wide range of tasks, including text generation, audio understanding.
    """

    END_OF_TEXT_TOKEN: str = "<|end_of_text|>"

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)
        self._device: str = get_torch_device_name()

    def _get_model(self, helm_model_name: str) -> LoadedGraniteSpeechProcessor:
        global _models_lock
        global _models

        model_name: str
        if helm_model_name == "granite-speech-3.3-8b":
            model_name = "ibm-granite/granite-speech-3.3-8b"
        elif helm_model_name == "granite-speech-3.3-2b":
            model_name = "ibm-granite/granite-speech-3.3-2b"
        elif helm_model_name == "granite-speech-3.2-8b":
            model_name = "ibm-granite/granite-speech-3.2-8b"
        else:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        # Ensure that only one thread is loading the model at a time
        with _models_lock:
            loaded_model_processor = _models[model_name]
            if loaded_model_processor is None:
                hlog(f"Loading model {model_name} and caching in memory...")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    device_map=self._device,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                ).eval()
                tokenizer = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
                generation_config = GenerationConfig.from_pretrained(model_name)

                _models[model_name] = LoadedGraniteSpeechProcessor(model, tokenizer, generation_config)
                loaded_model_processor = _models[model_name]

        assert loaded_model_processor is not None
        return loaded_model_processor

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        loaded_model_processor: LoadedGraniteSpeechProcessor = self._get_model(request.model_engine)
        model = loaded_model_processor.model
        tokenizer = loaded_model_processor.tokenizer
        generation_config = loaded_model_processor.generation_config

        input_query: List[Dict[str, Any]] = []
        prompt_text: str = ""

        input_query.append({"role": "system", "content": "You are a helpful AI assistant."})
        prompt_text += "<|start_of_role|>system<|end_of_role|>You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>"
        list_audio_urls: List[str] = []
        list_texts: List[str] = []
        for media_num, media_object in enumerate(request.multimodal_prompt.media_objects):
            if media_object.is_type("audio") and media_object.location:
                assert media_object.is_local_file, "Only local audio files are supported"
                list_audio_urls.append(media_object.location)
                list_texts.append("<|audio|>")
                prompt_text += "<|audio|>"
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                list_texts.append(media_object.text)
                prompt_text += media_object.text
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")
        prompt_text += "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"

        input_query.append({"role": "user", "content": "".join(list_texts)})
        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {prompt_text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        if request.model_engine == "granite-speech-3.2-8b":
                            inputs = tokenizer.tokenizer.apply_chat_template(
                                input_query, add_generation_prompt=True, tokenize=False
                            )
                            audios: List[Any] = []
                            for audio_url in list_audio_urls:
                                audios.append(torchaudio.load(audio_url, normalize=True)[0])
                            inputs = tokenizer(
                                text=inputs,
                                audios=audios,
                                return_tensors="pt",
                            ).to(self._device)
                        else:
                            inputs = tokenizer.apply_chat_template(
                                input_query, add_generation_prompt=True, tokenize=False
                            )
                            audios: List[Any] = []
                            for audio_url in list_audio_urls:
                                audios.append(torchaudio.load(audio_url, normalize=True)[0])
                            inputs = tokenizer(
                                text=inputs,
                                audio=audios,
                                return_tensors="pt",
                            ).to(self._device)
                        input_length = inputs["input_ids"].shape[1]
                        # Granite Speech counts input into the max_length,
                        # so we need to add the length of the prompt
                        pred = model.generate(
                            **inputs,
                            max_new_tokens=request.max_tokens + input_length,
                            generation_config=generation_config,
                        )[:, input_length:]

                        completion = tokenizer.tokenizer.decode(
                            pred.cpu()[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # The processor of Granite Speech consists an AutoTokenizer and a WhisperFeatureExtractor
                        tokens: List[str] = tokenizer.tokenizer.tokenize(completion)
                        return {"output": (completion, tokens)}

                    # Include the prompt and model name in the cache key
                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "completion_index": completion_index,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            "max_tokens": request.max_tokens,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                except RuntimeError as model_error:
                    return RequestResult(
                        success=False, cached=False, error=str(model_error), completions=[], embedding=[]
                    )

                text, tokens = result["output"]
                hlog(f"Generated: {text}")

                # Tokenize truncated text to get the list of tokens
                completions.append(
                    GeneratedOutput(
                        text=text, logprob=0, tokens=[Token(text=str(token), logprob=0) for token in tokens]
                    )
                )

                request_time += result["request_time"]
                # Use the datetime from the first completion because that's when the request was fired
                request_datetime = request_datetime or result.get("request_datetime")
                all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
