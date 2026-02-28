import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _dtype_from_string(value: Optional[str]) -> Optional[torch.dtype]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key or key in {"auto", "default"}:
        return None
    if key.startswith("torch."):
        key = key.split(".", 1)[1]
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "single": torch.float32,
        "float": torch.float32,
    }
    return mapping.get(key)


def _dtype_to_display(dtype: torch.dtype) -> str:
    text = str(dtype)
    if text.startswith("torch."):
        return text.split(".", 1)[1]
    return text


def _resolve_dtype(requested: Optional[str] = None) -> torch.dtype:
    if isinstance(requested, torch.dtype):
        return requested
    manual = _dtype_from_string(requested)
    if manual is not None:
        return manual

    if requested not in (None, "", "auto", "default"):
        logger.warning("[model] Unknown dtype '%s'; falling back to auto detection", requested)

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0) or ""
        except Exception:
            device_name = ""
        name_lower = device_name.lower()
        if "h100" in name_lower or "a100" in name_lower:
            return torch.bfloat16
        return torch.float16

    return torch.float32

class Model:
    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str,
        n: int = 3,
        trust_remote_code: bool = True,
        dtype: Optional[str] = "auto",
        adapter_path: Optional[str] = None,
    ):
        seed_env = os.getenv("GEN_SEED") or os.getenv("SEED")
        try:
            self._seed = int(seed_env) if seed_env not in (None, "",) else None
        except ValueError:
            self._seed = None

        self._dtype = dtype or "auto"
        self._resolved_dtype = _resolve_dtype(self._dtype)
        self._trust_remote_code = trust_remote_code

        use_vllm = os.getenv("USE_VLLM", "1") != "0"
        force_vllm_flag = os.getenv("FORCE_VLLM", "0") not in (None, "", "0", "false", "False")

        self._use_vllm = False
        self._vllm = None
        self._vllm_sampling = None
        self._hf_model = None
        self.backend = None
        self._lora_request = None

        gpu_util_val = None
        swap_gb_val = None
        tp_val = None
        eager_flag = False
        disable_chunked_prefill_flag = False
        max_batched_tokens_val = None
        max_seqs_val = None

        if use_vllm:
            try:
                from vllm import LLM, SamplingParams

                llm_kwargs = {
                    "model": model_name,
                    "trust_remote_code": trust_remote_code,
                    "dtype": self._resolved_dtype,
                }
                if self._seed is not None:
                    llm_kwargs["seed"] = self._seed

                gpu_util = os.getenv("VLLM_GPU_UTILIZATION")
                swap_gb = os.getenv("VLLM_SWAP_SPACE_GB")
                tp_env = os.getenv("VLLM_TENSOR_PARALLEL")
                eager = os.getenv("VLLM_ENFORCE_EAGER")
                disable_chunked_prefill_raw = os.getenv("VLLM_DISABLE_CHUNKED_PREFILL", "")
                disable_chunked_prefill_flag = disable_chunked_prefill_raw.lower() in ("1", "true", "yes", "on")
                mnbt = os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS")
                max_num_seqs = os.getenv("VLLM_MAX_NUM_SEQS")
                max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")

                if gpu_util:
                    try:
                        gpu_util_val = float(gpu_util)
                        llm_kwargs["gpu_memory_utilization"] = gpu_util_val
                    except ValueError:
                        logger.warning("[model] Invalid VLLM_GPU_UTILIZATION=%r; ignoring", gpu_util)
                if swap_gb:
                    try:
                        swap_gb_val = int(swap_gb)
                        llm_kwargs["swap_space"] = swap_gb_val
                    except ValueError:
                        logger.warning("[model] Invalid VLLM_SWAP_SPACE_GB=%r; ignoring", swap_gb)
                if tp_env:
                    try:
                        tp_val = int(tp_env)
                        llm_kwargs["tensor_parallel_size"] = tp_val
                    except ValueError:
                        logger.warning("[model] Invalid VLLM_TENSOR_PARALLEL=%r; ignoring", tp_env)
                eager_flag = eager == "1"
                if eager_flag:
                    llm_kwargs["enforce_eager"] = True
                llm_kwargs["enable_chunked_prefill"] = not disable_chunked_prefill_flag
                if mnbt:
                    try:
                        max_batched_tokens_val = int(mnbt)
                        llm_kwargs["max_num_batched_tokens"] = max_batched_tokens_val
                    except ValueError:
                        logger.warning("[model] Invalid VLLM_MAX_NUM_BATCHED_TOKENS=%r; ignoring", mnbt)
                if max_num_seqs:
                    try:
                        max_seqs_val = int(max_num_seqs)
                        llm_kwargs["max_num_seqs"] = max_seqs_val
                    except ValueError:
                        logger.warning("[model] Invalid VLLM_MAX_NUM_SEQS=%r; ignoring", max_num_seqs)
                if max_model_len_env:
                    try:
                        llm_kwargs["max_model_len"] = int(max_model_len_env)
                        print(f"[model] vLLM: max_model_len={llm_kwargs['max_model_len']}")
                    except ValueError:
                        print("[model] warn: bad VLLM_MAX_MODEL_LEN; ignoring")

                dtype_display = _dtype_to_display(self._resolved_dtype)
                print(f"[model] vLLM dtype set to {dtype_display}")
                logger.info("[model] vLLM dtype set to %s", dtype_display)

                enable_lora = adapter_path is not None
                self._vllm = LLM(enable_lora=enable_lora, **llm_kwargs)
                if adapter_path:
                    from vllm.lora.request import LoRARequest

                    adapter_name = os.path.basename(os.path.normpath(adapter_path))
                    self._lora_request = LoRARequest(
                        adapter_name,          # lora_name
                        1,                     # lora_int_id (any stable int is fine)
                        adapter_path,          # lora_local_path
                    )

                self._vllm_sampling = SamplingParams(
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=[stop] if stop else None,
                    seed=self._seed if self._seed is not None else None,
                )
                self._use_vllm = True
                self.backend = "vllm"
                logger.info("[model] LoRA active=%s", self._lora_request is not None)
            except Exception as exc:
                if force_vllm_flag:
                    raise RuntimeError(
                        f"vLLM init failed; refusing to fallback because FORCE_VLLM=1 ({exc})"
                    ) from exc
                logger.warning("[model] vLLM initialization failed (%s); falling back to HF", exc)
                self._use_vllm = False

        if not self._use_vllm:
            from transformers import AutoModelForCausalLM
            from peft import PeftModel

            torch_dtype = self._resolved_dtype
            dtype_display = _dtype_to_display(torch_dtype)
            print(f"[model] HF torch_dtype set to {dtype_display}")
            logger.info("[model] HF torch_dtype set to %s", dtype_display)

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            if adapter_path:
                print(f"[model] Loading LoRA adapter from {adapter_path}")
                self._hf_model = PeftModel.from_pretrained(self._hf_model, adapter_path)
            self.backend = "hf"
        else:
            self.backend = "vllm"

        if self.backend == "vllm":
            message = "[model] Backend: vLLM"
            print(message)
            logger.info(message)
            detail_msg = (
                "[model] vLLM: gpu_util=%s, swap_gb=%s, eager=%s, disable_chunked_prefill=%s, "
                "max_batched_tokens=%s, max_seqs=%s"
            )
            logger.info(
                detail_msg,
                gpu_util_val,
                swap_gb_val,
                eager_flag,
                disable_chunked_prefill_flag,
                max_batched_tokens_val,
                max_seqs_val,
            )
            print(
                detail_msg
                % (
                    gpu_util_val,
                    swap_gb_val,
                    eager_flag,
                    disable_chunked_prefill_flag,
                    max_batched_tokens_val,
                    max_seqs_val,
                )
            )
        else:
            message = "[model] Backend: HF"
            print(message)
            logger.info(message)
            fallback_msg = "[model] Using HF generate(); consider lowering batch size or enabling vLLM."
            print(fallback_msg)
            logger.info(fallback_msg)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat_template = getattr(self.tokenizer, "chat_template", None)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._stop = stop
        self._n = n

    def _sanitize_prompt(self, p: str) -> str:
        """
        Return a minimal safe prompt string for empty/invalid inputs.
        """
        if not p or not p.strip():
            return " "
        toks = self.tokenizer(p, add_special_tokens=False).input_ids
        if len(toks) == 0:
            return " "
        return p

    def _build_prompt(self, messages: List[Dict]) -> str:
        """
        Build a single string prompt from chat-style messages.
        Prefer tokenizer's chat_template; fall back to a plain-text format if missing.
        """
        if getattr(self.tokenizer, "apply_chat_template", None) and self.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: simple role-tagged conversation + assistant cue
        parts = []
        for m in messages:
            role = m.get("role", "").upper() or "USER"
            parts.append(f"{role}: {m.get('content','')}")
        parts.append("ASSISTANT:")  # cue for generation
        return "\n".join(parts)

    def generate(self, messages: List[Dict], n_override: Optional[int] = None) -> List[str]:
        """
        Generate text for a single chat-style prompt.
        Supports optional n_override to control sample count dynamically.
        """
        if n_override is not None:
            return self.generate_batch([messages], n_override=n_override)[0]
        return self.generate_batch([messages])[0]

    def generate_batch(
        self,
        batch_messages: List[List[Dict]],
        n_override: Optional[int] = None,
    ) -> List[List[str]]:
        prompts = [self._sanitize_prompt(self._build_prompt(msgs)) for msgs in batch_messages]
        n_generate = n_override if n_override is not None else self._n
        if n_generate <= 0:
            raise ValueError("n_override must be positive")  # defensive

        if self._use_vllm and self._vllm is not None and self._vllm_sampling is not None:
            sampling = self._vllm_sampling
            if n_generate != self._n:
                from vllm import SamplingParams
                sampling = SamplingParams(
                    n=n_generate,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    stop=[self._stop] if self._stop else None,
                    seed=self._seed if self._seed is not None else None,
                )
            # Guard rails for vLLM batch inputs.
            logger.info("[model] Incoming prompts=%d", len(prompts))
            safe_prompts = []
            index_map = []
            for i, p in enumerate(prompts):
                if not p or not p.strip():
                    logger.warning("[model] Dropping empty prompt at batch idx %d", i)
                    continue

                # Token-level check.
                toks = self.tokenizer(p, add_special_tokens=False).input_ids
                logger.debug("[model] prompt idx=%d token_len=%d", i, len(toks))
                if len(toks) == 0:
                    logger.warning("[model] Dropping zero-token prompt at batch idx %d", i)
                    continue

                safe_prompts.append(p)
                index_map.append(i)

            if not safe_prompts:
                return [[] for _ in prompts]

            logger.info("[model] vLLM batch size=%d (after filtering)", len(safe_prompts))
            assert len(safe_prompts) == len(index_map), "vLLM index map mismatch"

            try:
                outputs = self._vllm.generate(
                    safe_prompts,
                    sampling,
                    lora_request=self._lora_request,
                )
            except RuntimeError as e:
                logger.error("[model] vLLM generate failed, batch size=%d", len(safe_prompts))
                logger.error("[model] Falling back to HF for this batch only")
                logger.error("[model] vLLM failed; prompt indices=%s", index_map)
                raise

            final = [[] for _ in prompts]
            for out, idx in zip(outputs, index_map):
                final[idx] = [c.text for c in out.outputs]
            return final

        if self._hf_model is None:
            raise RuntimeError("HF model is not initialized.")

        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded.pop("token_type_ids", None)
        model_device = getattr(self._hf_model, "device", None)
        if model_device is not None:
            encoded = {k: v.to(model_device) for k, v in encoded.items()}

        prompt_lengths = encoded["attention_mask"].sum(dim=1)

        kwargs = {
            "max_new_tokens": self._max_tokens,
            "do_sample": True,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "num_return_sequences": n_generate,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self._stop:
            stop_token_ids = self.tokenizer.encode(self._stop)
            if stop_token_ids:
                kwargs["eos_token_id"] = stop_token_ids[-1]

        with torch.no_grad():
            outputs = self._hf_model.generate(**encoded, **kwargs)

        outputs = outputs.to("cpu")

        grouped: List[List[str]] = []
        for idx in range(len(prompts)):
            start = idx * n_generate
            end = start + n_generate
            samples = []
            prompt_len = prompt_lengths[idx].item()
            for row in outputs[start:end]:
                gen_tokens = row[int(prompt_len):]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if self._stop and self._stop in text:
                    text = text.split(self._stop, 1)[0]
                samples.append(text.strip())
            grouped.append(samples)
        return grouped
