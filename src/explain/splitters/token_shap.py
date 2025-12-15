from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

WHITESPACE = {" ", "\t", "\n", "\r"}
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB
SHARD_PATTERN = re.compile(r"^(?P<prefix>.+)_shard(?P<index>\d+)of(?P<total>\d+)$")

_DEBUG_FLAG = os.environ.get("SPLIT_TOKEN_SHAP_DEBUG")
_DEBUG_ENABLED = _DEBUG_FLAG not in (None, "", "0", "false", "False")

OUTPUT_FORMAT_JSON = "json"
OUTPUT_FORMAT_PICKLE = "pickle"
VALID_OUTPUT_FORMATS = {
    OUTPUT_FORMAT_JSON,
    OUTPUT_FORMAT_PICKLE,
}


def _debug(message: str) -> None:
    if _DEBUG_ENABLED:
        print(message, file=sys.stderr, flush=True)


def _skip_whitespace(buffer: str, position: int) -> int:
    length = len(buffer)
    while position < length and buffer[position] in WHITESPACE:
        position += 1
    return position


def _iter_json_records(
    path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handler:
        buffer = ""
        position = 0
        in_array = False

        while True:
            chunk = handler.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            position = _skip_whitespace(buffer, position)

            if not in_array:
                if position >= len(buffer):
                    buffer = ""
                    position = 0
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected '[' at start of records array in {path}")
                position += 1
                in_array = True

            while True:
                position = _skip_whitespace(buffer, position)
                if position >= len(buffer):
                    buffer = buffer[position:]
                    position = 0
                    break

                char = buffer[position]
                if char == "]":
                    return
                if char == ",":
                    position += 1
                    continue

                try:
                    record, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    buffer = buffer[position:]
                    position = 0
                    break

                if not isinstance(record, Mapping):
                    raise TypeError(
                        f"TokenSHAP record must be an object, got {type(record)!r}"
                    )

                yield dict(record)
                position = end

                if position > 1_000_000:
                    buffer = buffer[position:]
                    position = 0
                    break

        if not in_array:
            return

        while True:
            position = _skip_whitespace(buffer, position)
            if position >= len(buffer):
                break
            char = buffer[position]
            if char == "]":
                position += 1
                break
            if char == ",":
                position += 1
                continue
            record, end = decoder.raw_decode(buffer, position)
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"TokenSHAP record must be an object, got {type(record)!r}"
                )
            yield dict(record)
            position = end

        if buffer[position:].strip():
            raise ValueError(f"Unexpected trailing content in {path}")


class _RecordWriter:
    __slots__ = (
        "output_dir",
        "overwrite",
        "written",
        "processed",
        "offset",
        "output_format",
        "extension",
    )

    def __init__(
        self,
        output_dir: Path,
        overwrite: bool,
        *,
        offset: int = 0,
        output_format: str = OUTPUT_FORMAT_PICKLE,
    ) -> None:
        if output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite
        self.written = 0
        self.processed = 0
        self.offset = offset
        self.output_format = output_format
        self.extension = ".pkl" if output_format == OUTPUT_FORMAT_PICKLE else ".json"

    def __call__(self, record: Mapping[str, Any]) -> None:
        self.processed += 1
        graph_index_raw = record.get("graph_index", self.processed - 1 + self.offset)
        try:
            graph_index = int(graph_index_raw)
        except (TypeError, ValueError):
            graph_index = self.processed - 1 + self.offset

        destination = self.output_dir / f"graph_{graph_index:05d}{self.extension}"
        if destination.exists() and not self.overwrite:
            _debug(f"Skipping existing record: {destination}")
            return

        payload: Dict[str, Any]
        if isinstance(record, dict):
            payload = record
        else:
            payload = dict(record)

        if "global_graph_index" not in payload:
            payload["global_graph_index"] = graph_index

        if self.output_format == OUTPUT_FORMAT_PICKLE:
            with destination.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

        self.written += 1

        if isinstance(payload, dict):
            payload.clear()


def _infer_offset(path: Path) -> int:
    match = SHARD_PATTERN.match(path.stem)
    if not match:
        return 0
    try:
        shard_index = int(match.group("index"))
        total = int(match.group("total"))
    except ValueError:
        return 0
    if shard_index == 0 or total <= shard_index:
        return 0
    # Without knowing the shard strategy we cannot infer an exact offset.
    return 0


def split_records(
    input_path: Path,
    output_dir: Optional[Path],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    output_format: str = OUTPUT_FORMAT_PICKLE,
    limit: Optional[int] = None,
) -> int:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"LLM shard not found: {input_path}")
    if not input_path.suffix.lower().endswith("json"):
        raise ValueError(
            f"Unsupported TokenSHAP input format: {input_path.suffix}. Expected JSON."
        )

    if output_dir is None:
        output_dir = input_path.with_name(f"{input_path.stem}_split_{output_format}")
    output_dir = output_dir.expanduser().resolve()
    writer = _RecordWriter(
        output_dir,
        overwrite,
        offset=_infer_offset(input_path),
        output_format=output_format,
    )

    _debug(f"Splitting TokenSHAP records from '{input_path}' into '{output_dir}'")

    for record in _iter_json_records(input_path, chunk_size=chunk_size):
        writer(record)
        if limit and writer.processed >= limit:
            break

    _debug(
        f"Split completed: processed={writer.processed} written={writer.written} "
        f"output='{output_dir}'"
    )
    return writer.written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split TokenSHAP JSON shards into per-record pickle files."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON shard.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for per-record files. Defaults to <input>_split_<format>.",
    )
    parser.add_argument(
        "--format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=OUTPUT_FORMAT_PICKLE,
        help="Output format for individual records.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Streaming chunk size in bytes (default: 1 MiB).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-record files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N records (for debugging).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    split_records(
        args.input,
        args.output,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        output_format=args.format,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
