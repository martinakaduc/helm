# mypy: check_untyped_defs = False
from dataclasses import replace
from typing import Any, cast, Dict, List, Optional, Union

from helm.benchmark.model_metadata_registry import is_vlm
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt, truncate_sequence
from helm.common import multimodal_request_utils
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.media_object import TEXT_TYPE
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import GeneratedOutput, Request, RequestResult, Token, wrap_request_time
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.tokenizers.tokenizer import Tokenizer

try:
    import openai
    from openai import AzureOpenAI
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai", "azure-identity", "azure-mgmt-compute"])


class AzureOpenAIClient(CachingClient):
    END_OF_TEXT: str = "<|endoftext|>"

    # Error OpenAI throws when the image in the prompt violates their content policy
    INAPPROPRIATE_IMAGE_ERROR: str = "Your input image may contain content that is not allowed by our safety system"

    # Set the finish reason to this if the prompt violates OpenAI's content policy
    CONTENT_POLICY_VIOLATED_FINISH_REASON: str = (
        "The prompt violates Azure OpenAI's content management policy. "
        "See https://go.microsoft.com/fwlink/?linkid=2198766 for more information."
    )

    INAPPROPRIATE_TEXT_ERROR: str = (
        "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
    )

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.client = AzureOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version="2024-05-01-preview",
        )

    def _get_model_for_request(self, request: Request) -> str:
        return request.model_engine

    def _get_cache_key(self, raw_request: Dict, request: Request):
        cache_key = CachingClient.make_cache_key(raw_request, request)
        if request.multimodal_prompt:
            prompt_key: str = generate_uid_for_multimodal_prompt(request.multimodal_prompt)
            cache_key = {**cache_key, "multimodal_prompt": prompt_key}
            del cache_key["messages"]
        return cache_key

    def _make_embedding_request(self, request: Request) -> RequestResult:
        raw_request: Dict[str, Any]
        raw_request = {
            "input": request.prompt,
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
        }

        def do_it() -> Dict[str, Any]:
            return self.client.embeddings.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # If the user is requesting completions instead of an embedding, then `completions`
        # needs to be populated, and `embedding` should be an empty list and vice-versa.
        embedding: List[float] = []
        # If the user is requesting an embedding instead of completion
        # then completions would be left as an empty list. The embedding needs to be set.
        embedding = response["data"][0]["embedding"]

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=[],
            embedding=embedding,
        )

    def _make_chat_request(self, request: Request) -> RequestResult:
        messages: Optional[List[Dict[str, Union[str, Any]]]] = request.messages
        if (
            (request.prompt and request.messages)
            or (request.prompt and request.multimodal_prompt)
            or (request.messages and request.multimodal_prompt)
        ):
            raise ValueError(
                f"More than one of `prompt`, `messages` and `multimodal_prompt` was set in request: {request}"
            )
        if request.messages is not None:
            # Checks that all messages have a role and some content
            for message in request.messages:
                if not message.get("role") or not message.get("content"):
                    raise ValueError("All messages must have a role and content")
            # Checks that the last role is "user"
            if request.messages[-1]["role"] != "user":
                raise ValueError("Last message must have role 'user'")
            if request.prompt != "":
                hlog("WARNING: Since message is set, prompt will be ignored")
        else:
            # Convert prompt into a single message
            # For now, put the whole prompt in a single user message, and expect the response
            # to be returned in a single assistant message.
            # TODO: Support ChatML for creating multiple messages with different roles.
            # See: https://github.com/openai/openai-python/blob/main/chatml.md

            # Content can either be text or a list of multimodal content made up of text and images:
            # https://platform.openai.com/docs/guides/vision
            content: Union[str, List[Union[str, Any]]]
            if request.multimodal_prompt is not None:
                content = []
                request.validate()
                for media_object in request.multimodal_prompt.media_objects:
                    if media_object.is_type("image") and media_object.location:
                        from helm.common.images_utils import encode_base64

                        base64_image: str = encode_base64(media_object.location)
                        image_object: Dict[str, str] = {"url": f"data:image/jpeg;base64,{base64_image}"}
                        content.append({"type": "image_url", "image_url": image_object})
                    elif media_object.is_type("audio") and media_object.location:
                        base64_audio: str = multimodal_request_utils.get_contents_as_base64(media_object.location)
                        format: str = media_object.content_type.split("/")[1]
                        if format == "mpeg":
                            # OpenAI expects "mp3" for mpeg audio
                            format = "mp3"

                        content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {"data": base64_audio, "format": format},
                            }
                        )
                    elif media_object.is_type(TEXT_TYPE):
                        content.append({"type": media_object.type, "text": media_object.text})
                    else:
                        raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

            else:
                content = request.prompt

            messages = [{"role": "user", "content": content}]

        raw_request: Dict[str, Any] = {
            "model": self._get_model_for_request(request),
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            # Note: Chat models may require adding an extra token to max_tokens
            # for the internal special role token.
            "max_tokens": request.max_tokens,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        # OpenAI's vision API doesn't allow None values for stop.
        # Fails with "body -> stop: none is not an allowed value" if None is passed.
        if is_vlm(request.model) and raw_request["stop"] is None:
            raw_request.pop("stop")

        # Special handling for o1 models.
        # Refer to the "Reasoning models" documentation further discussion of o1 model limitations:
        # https://platform.openai.com/docs/guides/reasoning
        if request.model_engine.startswith("o1"):
            # Avoid error:
            # "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."  # noqa: E501
            # Note that openai>=1.45 is needed for this
            if raw_request["max_tokens"]:
                raw_request["max_completion_tokens"] = raw_request["max_tokens"]
                raw_request.pop("max_tokens")
            # Avoid error:
            # "Invalid type for 'stop': expected an unsupported value, but got null instead."
            if raw_request["stop"] is None:
                raw_request.pop("stop")

        # Special handling for gpt-4o-audio-preview
        # See: https://platform.openai.com/docs/guides/audio
        if request.model_engine.startswith("gpt-4o-audio-preview"):
            raw_request["modalities"] = ["text"]

            # Avoid error:
            # OpenAI error: Error code: 400 - {'error': {'message': "[{'type': 'string_type', 'loc': ('body', 'stop', 'str'), 'msg': 'Input should be a valid string', 'input': None}, {'type': 'list_type', 'loc': ('body', 'stop', 'list[str]'), 'msg': 'Input should be a valid list', 'input': None}, {'type': 'list_type', 'loc': ('body', 'stop', 'list[list[int]]'), 'msg': 'Input should be a valid list', 'input': None}]", 'type': 'invalid_request_error', 'param': None, 'code': None}}  # noqa: 3501
            if raw_request["stop"] is None:
                raw_request.pop("stop")

        def do_it() -> Dict[str, Any]:
            return self.client.chat.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            if self.INAPPROPRIATE_IMAGE_ERROR in str(e):
                hlog(f"Failed safety check: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                return RequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    embedding=[],
                )
            if self.INAPPROPRIATE_TEXT_ERROR in str(e):
                hlog(f"Failed safety check: {str(request)}")
                empty_completion = GeneratedOutput(
                    text="",
                    logprob=0,
                    tokens=[],
                    finish_reason={"reason": self.CONTENT_POLICY_VIOLATED_FINISH_REASON},
                )
                return RequestResult(
                    success=True,
                    cached=False,
                    request_time=0,
                    completions=[empty_completion] * request.num_completions,
                    embedding=[],
                )
            error: str = f"AzureOpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            # The OpenAI chat completion API doesn't support echo.
            # If `echo_prompt` is true, combine the prompt and completion.
            raw_completion_content = raw_completion["message"]["content"]
            text: str = request.prompt + raw_completion_content if request.echo_prompt else raw_completion_content
            if text is None:
                text = ""
            # The OpenAI chat completion API doesn't return us tokens or logprobs, so we tokenize ourselves.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )
            # Log probs are not currently not supported by the OpenAI chat completion API, so set to 0 for now.
            tokens: List[Token] = [
                Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
            ]
            completion = GeneratedOutput(
                text=text,
                logprob=0,  # OpenAI does not provide logprobs
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            completions.append(truncate_sequence(completion, request))  # Truncate the text by stop sequences

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def _to_raw_completion_request(self, request: Request) -> Dict[str, Any]:
        raw_request: Dict[str, Any] = {
            # Note: In older deprecated versions of the OpenAI API, "model" used to be "engine".
            "model": self._get_model_for_request(request),
            "prompt": request.prompt,
            "temperature": request.temperature,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "best_of": request.top_k_per_token,
            "logprobs": request.top_k_per_token,
            "stop": request.stop_sequences or None,  # API doesn't like empty list
            "top_p": request.top_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "echo": request.echo_prompt,
        }

        # OpenAI doesn't let you ask for more completions than the number of
        # per-token candidates.
        raw_request["best_of"] = max(raw_request["best_of"], raw_request["n"])
        raw_request["logprobs"] = max(raw_request["logprobs"], raw_request["n"])

        return raw_request

    def _make_completion_request(self, request: Request) -> RequestResult:
        raw_request = self._to_raw_completion_request(request)

        def do_it() -> Dict[str, Any]:
            return self.client.completions.create(**raw_request).model_dump(mode="json")

        try:
            cache_key = self._get_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except openai.OpenAIError as e:
            error: str = f"OpenAI error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            sequence_logprob = 0
            tokens: List[Token] = []

            raw_data = raw_completion["logprobs"]
            for (
                text,
                logprob,
            ) in zip(raw_data["tokens"], raw_data["token_logprobs"]):
                tokens.append(Token(text=text, logprob=logprob or 0))
                sequence_logprob += logprob or 0
            completion = GeneratedOutput(
                text=raw_completion["text"],
                logprob=sequence_logprob,
                tokens=tokens,
                finish_reason={"reason": raw_completion["finish_reason"]},
            )
            # OpenAI sends us back tokens past the end of text token,
            # so we need to manually truncate the list of tokens.
            # TODO: filed an issue with their support to check what the expected behavior here is.
            completion = truncate_sequence(
                completion, replace(request, stop_sequences=request.stop_sequences + [AzureOpenAIClient.END_OF_TEXT])
            )
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def make_request(self, request: Request) -> RequestResult:
        if request.embedding:
            return self._make_embedding_request(request)
        else:
            return self._make_chat_request(request)


class AzureOpenAILegacyCompletionsClient(AzureOpenAIClient):
    def make_request(self, request: Request) -> RequestResult:
        return self._make_completion_request(request)