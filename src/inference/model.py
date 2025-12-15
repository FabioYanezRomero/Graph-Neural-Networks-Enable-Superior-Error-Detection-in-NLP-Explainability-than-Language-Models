from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None  # type: ignore


_DEFAULT_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("teacher_prompt.json")
_DEFAULT_MODEL_PATH = Path("./models/qwen3-14b")
_DEFAULT_OUTPUT_ROOT = Path("outputs/inference")
_DEFAULT_DATASET_DIR = Path("data")
_DEFAULT_MAX_NEW_TOKENS = 1024
_DEFAULT_DECISION_MAX_NEW_TOKENS = 64
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_TOP_P = 0.9
_DEFAULT_TOP_K = 50
_DEFAULT_REPETITION_PENALTY = 1.0
_DEFAULT_BATCH_SIZE = 1


class _SafeFormatDict(dict):
    """Return a readable placeholder when formatting misses a key."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - str.format behaviour
        return "{" + key + "}"


def _format_text(template: str, variables: Mapping[str, Any]) -> str:
    return template.format_map(_SafeFormatDict(**variables))


def _canonical_component(value: str) -> str:
    """Make a filesystem-friendly identifier."""
    safe = value.replace("/", "__").replace("\\", "__").strip()
    safe = safe.replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in safe)
    return safe or "unknown"


class BasePromptTemplate:
    def __init__(self, source: Path) -> None:
        self.source = source

    @property
    def is_chat(self) -> bool:
        raise NotImplementedError

    def format(self, example: Mapping[str, Any]) -> Any:
        raise NotImplementedError


class TextPromptTemplate(BasePromptTemplate):
    def __init__(self, source: Path) -> None:
        super().__init__(source)
        self.template = source.read_text(encoding="utf-8")

    @property
    def is_chat(self) -> bool:
        return False

    def format(self, example: Mapping[str, Any]) -> str:
        return _format_text(self.template, example)


class ChatPromptTemplate(BasePromptTemplate):
    def __init__(self, source: Path, messages: Sequence[Mapping[str, str]]) -> None:
        super().__init__(source)
        self.messages = list(messages)

    @property
    def is_chat(self) -> bool:
        return True

    def format(self, example: Mapping[str, Any]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        for message in self.messages:
            formatted.append(
                {
                    "role": message["role"],
                    "content": _format_text(message["content"], example),
                }
            )
        return formatted


def load_prompt_template(path: Path) -> BasePromptTemplate:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(m, Mapping) for m in data):
            raise ValueError(f"Chat prompt JSON must be a list of role/content mappings: {path}")
        for message in data:
            if "role" not in message or "content" not in message:
                raise ValueError(f"Each chat message requires 'role' and 'content': {path}")
        return ChatPromptTemplate(path, data)
    return TextPromptTemplate(path)


@dataclass
class QwenInferenceConfig:
    dataset_name: str
    split: str = "train"
    dataset_dir: Path = _DEFAULT_DATASET_DIR
    text_field: str = "text"
    prompt_path: Path = _DEFAULT_PROMPT_PATH
    reason_prompt_path: Optional[Path] = None
    model_path: Path = _DEFAULT_MODEL_PATH
    tokenizer_path: Optional[Path] = None
    output_dir: Path = _DEFAULT_OUTPUT_ROOT
    output_path: Optional[Path] = None
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    decision_max_new_tokens: int = _DEFAULT_DECISION_MAX_NEW_TOKENS
    temperature: float = _DEFAULT_TEMPERATURE
    top_p: float = _DEFAULT_TOP_P
    top_k: int = _DEFAULT_TOP_K
    repetition_penalty: float = _DEFAULT_REPETITION_PENALTY
    do_sample: Optional[bool] = None
    batch_size: int = _DEFAULT_BATCH_SIZE
    device: str = "auto"
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    offload_folder: Optional[Path] = None
    save_prompt: bool = False

    @property
    def resolved_decision_prompt_path(self) -> Path:
        return Path(self.prompt_path)

    @property
    def resolved_reason_prompt_path(self) -> Path:
        if self.reason_prompt_path is not None:
            return Path(self.reason_prompt_path)
        decision_path = self.resolved_decision_prompt_path
        candidate = decision_path.with_name(f"{decision_path.stem}_reasoning{decision_path.suffix}")
        return candidate if candidate.exists() else decision_path

    @property
    def resolved_model_path(self) -> Path:
        return Path(self.model_path)

    @property
    def resolved_tokenizer_path(self) -> Path:
        return Path(self.tokenizer_path or self.model_path)

    @property
    def resolved_output_path(self) -> Path:
        if self.output_path:
            return Path(self.output_path)
        dataset_part = _canonical_component(self.dataset_name)
        split_part = _canonical_component(self.split)
        model_part = _canonical_component(Path(self.model_path).name or "model")
        filename = f"{model_part}.jsonl"
        return Path(self.output_dir) / dataset_part / split_part / filename


def _load_local_records(dataset_dir: Path, split: str) -> List[Dict[str, Any]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Local dataset directory does not exist: {dataset_dir}")

    candidates = [
        dataset_dir / f"{split}.jsonl",
        dataset_dir / split / "data.jsonl",
        dataset_dir / split / f"{split}.jsonl",
        dataset_dir / f"{split}.json",
        dataset_dir / split / "data.json",
        dataset_dir / f"{split}.csv",
        dataset_dir / split / "data.csv",
        dataset_dir / f"{split}.txt",
        dataset_dir / split / "data.txt",
    ]

    target: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            target = candidate
            break

    if target is None:
        raise FileNotFoundError(
            f"Unable to locate split '{split}' under {dataset_dir}. Expected one of: "
            + ", ".join(str(c.relative_to(dataset_dir)) for c in candidates)
        )

    if target.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if target.suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload  # type: ignore[return-value]
        if isinstance(payload, dict) and split in payload:
            split_value = payload[split]
            if isinstance(split_value, list):
                return split_value  # type: ignore[return-value]
        raise ValueError(f"JSON dataset must be a list or contain key '{split}': {target}")
    if target.suffix == ".csv":
        return _read_csv(target)

    # Plain-text fallback: one example per line
    with target.open("r", encoding="utf-8") as fh:
        return [{"text": line.rstrip("\n")} for line in fh if line.strip()]


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = ","
        header = sample.splitlines()[0] if sample else ""
        if "|" in header and header.count("|") >= 1:
            delimiter = "|"
        else:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
                delimiter = dialect.delimiter
            except csv.Error:
                pass
        reader = csv.DictReader(fh, delimiter=delimiter)
        records: List[Dict[str, Any]] = []
        for row in reader:
            if None in row:
                row = {k: v for k, v in row.items() if k is not None}
            records.append(dict(row))
        return records


def _load_local_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload  # type: ignore[return-value]
        raise ValueError(f"JSON dataset file must contain a list of records: {path}")
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as fh:
            return [{"text": line.rstrip("\n")} for line in fh if line.strip()]

    raise ValueError(f"Unsupported dataset file type '{suffix}' for {path}")


def load_dataset_records(cfg: QwenInferenceConfig) -> List[Dict[str, Any]]:
    local_dir = Path(cfg.dataset_dir) / cfg.dataset_name
    if local_dir.exists():
        if local_dir.is_dir():
            return _load_local_records(local_dir, cfg.split)
        if local_dir.is_file():
            return _load_local_file(local_dir)

    file_candidates = [Path(cfg.dataset_dir) / cfg.dataset_name]
    if not Path(cfg.dataset_name).suffix:
        file_candidates.extend(
            Path(cfg.dataset_dir) / f"{cfg.dataset_name}.{ext}" for ext in ("csv", "jsonl", "json", "txt")
        )
    for candidate in file_candidates:
        if candidate.exists() and candidate.is_file():
            return _load_local_file(candidate)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets library required to pull remote dataset; install via `pip install datasets`."
        ) from exc

    dataset = load_dataset(cfg.dataset_name, split=cfg.split)
    return [dict(item) for item in dataset]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


class QwenTeacher:
    def __init__(self, cfg: QwenInferenceConfig) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.decision_prompt = load_prompt_template(cfg.resolved_decision_prompt_path)
        self.reason_prompt = load_prompt_template(cfg.resolved_reason_prompt_path)
        tokenizer_path = str(cfg.resolved_tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            eos = self.tokenizer.eos_token
            if eos is None:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            else:
                self.tokenizer.pad_token = eos

        model_path = str(cfg.resolved_model_path)
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.offload_folder is not None:
            model_kwargs["offload_folder"] = str(cfg.offload_folder)

        use_cuda = self.device.type == "cuda"
        quantized = False
        if cfg.load_in_4bit and BitsAndBytesConfig is None:
            print("bitsandbytes not available; loading full precision model instead of 4-bit quantization.")
        if cfg.load_in_4bit and BitsAndBytesConfig is not None and not use_cuda:
            print("CUDA device not available; loading full precision model instead of 4-bit quantization.")
        if cfg.load_in_4bit and BitsAndBytesConfig is not None and use_cuda:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            quantized = True
        else:
            if use_cuda:
                model_kwargs["torch_dtype"] = torch.float16
            else:
                model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if not quantized:
            self.model.to(self.device)

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.config, "eos_token_id", None) is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, record: Mapping[str, Any], template: BasePromptTemplate) -> str:
        context = dict(record)
        if self.cfg.text_field not in context:
            raise KeyError(
                f"Record missing expected text field '{self.cfg.text_field}'. "
                f"Available keys: {sorted(context.keys())}"
            )
        context.setdefault("input_text", context[self.cfg.text_field])
        context.setdefault("dataset_name", self.cfg.dataset_name)
        context.setdefault("split", self.cfg.split)

        if template.is_chat:
            messages = template.format(context)
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return template.format(context)

    def _generation_kwargs(self) -> Dict[str, Any]:
        do_sample = self.cfg.do_sample
        if do_sample is None:
            do_sample = self.cfg.temperature > 0
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "repetition_penalty": self.cfg.repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(
                {
                    "temperature": max(self.cfg.temperature, 1e-5),
                    "top_p": self.cfg.top_p,
                    "top_k": self.cfg.top_k,
                }
            )
        return kwargs

    def _generate_from_prompt(
        self,
        template: BasePromptTemplate,
        record: Mapping[str, Any],
        generation_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        prompt_text = self._build_prompt(record, template)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = output_ids[0, prompt_len:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return {"prompt": prompt_text, "response": generated_text}

    @staticmethod
    def _normalise_decision_label(raw: str) -> str:
        # Robust extractor:
        # - normalize whitespace/case
        # - strip simple markdown/punctuation wrappers
        # - choose earliest occurrence among allow/dismiss
        text = raw if isinstance(raw, str) else str(raw)
        cleaned = re.sub(r"\s+", " ", text.lower()).strip()

        # Quick exact or prefix match
        for label in ("allow", "dismiss"):
            if cleaned == label or cleaned.startswith(label + " "):
                return label

        # Remove trivial punctuation around words
        scrubbed = re.sub(r"[`*_\-~#>]+", " ", cleaned)

        # Find earliest occurrence
        first_pos = None
        first_label = None
        for label in ("allow", "dismiss"):
            # word boundary to avoid matching inside longer words
            m = re.search(rf"\b{label}\b", scrubbed)
            pos = m.start() if m else -1
            if pos != -1 and (first_pos is None or pos < first_pos):
                first_pos = pos
                first_label = label
        if first_label is not None:
            return first_label

        # Fallback: last resort search without word boundaries
        for label in ("allow", "dismiss"):
            pos = scrubbed.find(label)
            if pos != -1:
                return label

        return "error"

    def generate(self, records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        # Build separate generation kwargs for decision (short) and reasoning (unlimited)
        decision_kwargs = self._generation_kwargs()
        decision_kwargs = dict(decision_kwargs)
        # Make decision greedy and short
        decision_kwargs["do_sample"] = False
        decision_kwargs.pop("temperature", None)
        decision_kwargs.pop("top_p", None)
        decision_kwargs.pop("top_k", None)
        decision_kwargs["max_new_tokens"] = max(1, int(self.cfg.decision_max_new_tokens))

        reasoning_kwargs = self._generation_kwargs()
        reasoning_kwargs = dict(reasoning_kwargs)
        # Ensure sampling enabled for reasoning when temperature > 0
        if self.cfg.do_sample is False:
            # If globally forced greedy, keep greedy; otherwise enable sampling from temperature
            pass
        # Cap reasoning length to a large but finite budget
        reasoning_kwargs["max_new_tokens"] = 1024
        for idx, record in enumerate(tqdm(records, desc="Generating", unit="example")):
            decision_step = self._generate_from_prompt(self.decision_prompt, record, decision_kwargs)
            decision_label = self._normalise_decision_label(decision_step["response"])

            reason_context = dict(record)
            reason_context["predicted_decision_label"] = decision_label
            reason_step = self._generate_from_prompt(self.reason_prompt, reason_context, reasoning_kwargs)

            example_output = {
                "dataset": self.cfg.dataset_name,
                "split": self.cfg.split,
                "model": str(self.cfg.model_path),
                "index": idx,
                "prompt": None,
                "response": {
                    "decision_label_raw": decision_step["response"],
                    "decision_label": decision_label,
                    "reasoning": reason_step["response"],
                },
                "input": dict(record),
            }
            if self.cfg.save_prompt:
                example_output["prompt"] = {
                    "decision": decision_step["prompt"],
                    "reasoning": reason_step["prompt"],
                }
            outputs.append(example_output)
        return outputs

    def run(self) -> Path:
        records = load_dataset_records(self.cfg)
        outputs = self.generate(records)
        output_path = self.cfg.resolved_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for item in outputs:
                serialisable = {k: v for k, v in item.items() if k != "prompt" or self.cfg.save_prompt}
                fh.write(json.dumps(serialisable, ensure_ascii=False, default=str) + "\n")
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3 14B teacher inference")
    parser.add_argument("--dataset_name", required=True, help="Name of dataset directory or HF dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--prompt_path", default=str(_DEFAULT_PROMPT_PATH))
    parser.add_argument(
        "--reason_prompt_path",
        default=None,
        help="Optional path to the reasoning prompt used for the second-generation step",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--decision_max_new_tokens",
        type=int,
        default=_DEFAULT_DECISION_MAX_NEW_TOKENS,
        help="Max new tokens for the first (decision) step; keep small (e.g., 2)",
    )
    parser.add_argument("--temperature", type=float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=_DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("--repetition_penalty", type=float, default=_DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--force_sample", action="store_true", help="Force sampling even if temperature is zero")
    parser.add_argument("--force_greedy", action="store_true", help="Disable sampling regardless of temperature")
    parser.add_argument("--batch_size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable_4bit", action="store_true", help="Load the model without 4-bit quantization")
    parser.add_argument("--dataset_dir", default=str(_DEFAULT_DATASET_DIR))
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true", help="Allow loading custom model code")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false", help="Disable custom model code")
    parser.set_defaults(trust_remote_code=True)
    return parser


def parse_args_to_config(args: argparse.Namespace) -> QwenInferenceConfig:
    do_sample: Optional[bool]
    if args.force_sample and args.force_greedy:
        raise ValueError("--force_sample and --force_greedy are mutually exclusive")
    if args.force_sample:
        do_sample = True
    elif args.force_greedy:
        do_sample = False
    else:
        do_sample = None

    load_in_4bit = not args.disable_4bit

    cfg = QwenInferenceConfig(
        dataset_name=args.dataset_name,
        split=args.split,
        dataset_dir=Path(args.dataset_dir),
        text_field=args.text_field,
        prompt_path=Path(args.prompt_path),
        reason_prompt_path=Path(args.reason_prompt_path) if args.reason_prompt_path else None,
        model_path=Path(args.model_path),
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        output_dir=Path(args.output_dir),
        output_path=Path(args.output_path) if args.output_path else None,
        max_new_tokens=args.max_new_tokens,
        decision_max_new_tokens=args.decision_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=do_sample,
        batch_size=args.batch_size,
        device=args.device,
        load_in_4bit=load_in_4bit,
        trust_remote_code=args.trust_remote_code,
        offload_folder=Path(args.offload_folder) if args.offload_folder else None,
        save_prompt=args.save_prompt,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = parse_args_to_config(args)
    teacher = QwenTeacher(cfg)
    return teacher.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
