from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.append(str(ROOT / "code"))

from project_paths import PROCESSED_DATA_DIR, SENTENCE_MODEL_NAME_OR_PATH


DEFAULT_INPUT_DIR = PROCESSED_DATA_DIR / "text_features"
DEFAULT_OUTPUT_DIR = PROCESSED_DATA_DIR / "text_features_enriched"
TIMESTEPS = 10

INTENT_LEXICON = {
    "menu": {
        "english": "menu open menu show menu options",
        "chinese": ["菜单", "目錄", "目录", "选项", "界面"],
        "pinyin": ["cai dan", "caidan", "xuan xiang", "xuanxiang"],
    },
    "select": {
        "english": "select choose confirm pick this",
        "chinese": ["选择", "选中", "确定", "确认", "这个"],
        "pinyin": ["xuan ze", "xuanze", "xuan zhong", "xuanzhong", "que ding", "queding"],
    },
    "magnify": {
        "english": "magnify zoom in enlarge bigger",
        "chinese": ["放大", "变大", "拉近", "扩大"],
        "pinyin": ["fang da", "fangda", "bian da", "bianda", "la jin", "lajin"],
    },
    "narrow": {
        "english": "narrow zoom out shrink smaller reduce",
        "chinese": ["缩小", "变小", "拉远", "减小"],
        "pinyin": ["suo xiao", "suoxiao", "bian xiao", "bianxiao", "la yuan", "layuan"],
    },
    "brush": {
        "english": "brush erase wipe clean draw stroke",
        "chinese": ["刷", "擦", "清除", "画", "涂", "笔"],
        "pinyin": ["shua", "ca", "qing chu", "qingchu", "hua", "tu", "bi"],
    },
    "cancel": {
        "english": "cancel stop close exit back undo",
        "chinese": ["取消", "停止", "关闭", "退出", "返回", "撤销"],
        "pinyin": ["qu xiao", "quxiao", "ting zhi", "tingzhi", "guan bi", "guanbi", "tui chu", "tuichu"],
    },
}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def compact_pinyin(value: str) -> str:
    return re.sub(r"\s+", "", normalize_space(value))


def find_intent_hints(text: str, pinyin: str) -> list[str]:
    hints: list[str] = []
    text_norm = text.strip()
    pinyin_norm = normalize_space(pinyin)
    pinyin_compact = compact_pinyin(pinyin)
    for intent, lexicon in INTENT_LEXICON.items():
        matched = False
        for token in lexicon["chinese"]:
            if token and token in text_norm:
                matched = True
        for token in lexicon["pinyin"]:
            token_norm = normalize_space(token)
            token_compact = compact_pinyin(token)
            if token_norm and token_norm in pinyin_norm:
                matched = True
            if token_compact and token_compact in pinyin_compact:
                matched = True
        if matched:
            hints.append(intent)
    return hints


def build_enriched_text(segment: dict[str, object], include_vocab: bool) -> str:
    text = str(segment.get("text", "") or "").strip()
    pinyin = str(segment.get("pinyin", "") or "").strip()
    hints = find_intent_hints(text, pinyin)
    hint_text = " ".join(f"intent_hint_{item}" for item in hints) if hints else "intent_hint_unknown"
    fields = [
        f"asr_chinese: {text}" if text else "asr_chinese: empty",
        f"asr_pinyin: {pinyin}" if pinyin else "asr_pinyin: empty",
        f"matched_intents: {hint_text}",
    ]
    if include_vocab:
        vocab = " ; ".join(
            f"{intent}: {lexicon['english']}"
            for intent, lexicon in INTENT_LEXICON.items()
        )
        fields.append(f"intent_vocabulary: {vocab}")
    return " | ".join(fields)


def process_file(path: Path, output_dir: Path, model: SentenceTransformer, include_vocab: bool) -> None:
    with path.open("r", encoding="utf-8") as file:
        segments = json.load(file)
    enriched_texts = [build_enriched_text(segment, include_vocab) for segment in segments]
    embeddings = model.encode(enriched_texts, normalize_embeddings=True, show_progress_bar=False)
    features = np.tile(np.asarray(embeddings, dtype=np.float32)[:, np.newaxis, :], (1, TIMESTEPS, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / path.name.replace(".json", ".npy")
    json_path = output_dir / path.name
    out_segments = []
    for segment, enriched in zip(segments, enriched_texts):
        record = dict(segment)
        record["enriched_text"] = enriched
        record["intent_hints"] = find_intent_hints(str(record.get("text", "") or ""), str(record.get("pinyin", "") or ""))
        out_segments.append(record)

    np.save(npy_path, {"features": features, "metadata": out_segments})
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(out_segments, file, indent=2, ensure_ascii=False)
    print(f"[saved] {npy_path} shape={features.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched ASR text embeddings from cached ASR JSON files.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=str(SENTENCE_MODEL_NAME_OR_PATH))
    parser.add_argument("--no-vocab", action="store_true", help="Do not append the shared intent vocabulary text.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model)
    model_source = str(model_path) if model_path.exists() else args.model
    model = SentenceTransformer(model_source)

    json_files = sorted(input_dir.glob("text_features_*.json"))
    if args.limit > 0:
        json_files = json_files[: args.limit]
    if not json_files:
        raise SystemExit(f"No text feature JSON files found under: {input_dir}")

    print(f"[text-enhance] files={len(json_files)} input={input_dir} output={output_dir}")
    for path in json_files:
        process_file(path, output_dir, model, include_vocab=not args.no_vocab)


if __name__ == "__main__":
    main()
