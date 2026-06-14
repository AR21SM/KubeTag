from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Any

TRIAGE_COMMANDS = {
    "approve",
    "area",
    "assign",
    "cc",
    "cherrypick",
    "close",
    "help",
    "hold",
    "hold-cancel",
    "joke",
    "kind",
    "label",
    "lgtm",
    "meow",
    "milestone",
    "ok-to-test",
    "override",
    "pony",
    "priority",
    "reopen",
    "release-note",
    "release-note-action-required",
    "release-note-none",
    "remove-area",
    "remove-cherrypick",
    "remove-kind",
    "remove-label",
    "remove-milestone",
    "remove-needs-rebase",
    "remove-priority",
    "remove-sig",
    "retest",
    "retest-required",
    "shrug",
    "sig",
    "skip",
    "test",
    "this-is-fine",
    "triage",
    "unassign",
    "uncc",
    "unhold",
    "woof",
}

PREPROCESSING_VERSION = "kubetag-text-v3"

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")
_RAW_URL_RE = re.compile(r"https?://[^\s<>()\]]+", re.I)
_ISSUE_REF_RE = re.compile(r"(?<![A-Za-z0-9])#\d+\b")
_HTML_BR_RE = re.compile(r"(?i)<br\s*/?>")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_EMPTY_BRACKETS_RE = re.compile(r"\[\s*\]|\(\s*\)")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def _patterns(
    known_labels: Iterable[str],
) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    command_names = "|".join(
        sorted(
            (re.escape(command) for command in TRIAGE_COMMANDS), key=len, reverse=True
        )
    )
    command_re = re.compile(
        r"(?im)^[ \t]*(?:>[ \t]*)*"
        r"(?:(?:[-*+]|\d+[.)])[ \t]+(?:\[[ xX]\][ \t]+)?)?"
        r"`{0,3}[ \t]*/(?:" + command_names + r")\b[^\n]*(?:\n|$)"
    )
    backticked_command_re = re.compile(
        r"`+[ \t]*/(?:" + command_names + r")\b[^`\r\n]*`+",
        re.IGNORECASE,
    )
    label_patterns: list[str] = []
    for label in sorted(set(known_labels), key=len, reverse=True):
        taxonomy, name = label.split("/", 1)
        label_patterns.append(
            rf"(?<![A-Za-z0-9])(?:{re.escape(taxonomy)}/{re.escape(name)}|"
            rf"{re.escape(taxonomy)}%2[fF]{re.escape(name)}|"
            rf"{re.escape(taxonomy)}[-_:]{re.escape(name)})(?![A-Za-z0-9])"
        )
        if taxonomy == "sig":
            label_patterns.append(
                rf"(?<![A-Za-z0-9])@?kubernetes/sig-{re.escape(name)}"
                rf"(?:-[a-z0-9-]+)?(?![A-Za-z0-9])"
            )
    label_expression = "|".join(f"(?:{part})" for part in label_patterns) or r"(?!)"
    label_re = re.compile(label_expression, re.I)
    return command_re, backticked_command_re, label_re


def clean_issue_field(text: str | None, known_labels: Iterable[str]) -> str:
    if not text:
        return ""
    command_re, backticked_command_re, label_re = _patterns(known_labels)
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    value = _HTML_COMMENT_RE.sub(" ", value)
    value = command_re.sub("\n", value)
    value = backticked_command_re.sub(" ", value)
    value = label_re.sub(" ", value)
    value = _MD_IMAGE_RE.sub(lambda match: match.group(1).strip(), value)
    value = _MD_LINK_RE.sub(lambda match: match.group(1).strip(), value)
    value = _HTML_BR_RE.sub("\n", value)
    value = _HTML_TAG_RE.sub(" ", value)
    value = _RAW_URL_RE.sub("<URL>", value)
    value = _ISSUE_REF_RE.sub("<ISSUE_REF>", value)
    value = _EMPTY_BRACKETS_RE.sub(" ", value)
    value = "".join(
        character if character in {"\n", "\t"} or ord(character) >= 32 else " "
        for character in value
    )
    lines = []
    for line in value.split("\n"):
        normalized = _MULTI_SPACE_RE.sub(" ", line).strip()
        if normalized:
            lines.append(normalized)
    return _MANY_NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def prepare_text(
    title: str,
    body: str | None,
    known_labels: Iterable[str] = (),
) -> str:
    clean_title = clean_issue_field(title, known_labels)
    clean_body = clean_issue_field(body, known_labels)
    return f"Title: {clean_title}\nBody: {clean_body}".rstrip()


def encode_head_tail(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
) -> dict[str, Any]:
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    content_limit = max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
    tokenized = tokenizer(
        [str(text) for text in texts],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    rows = []
    for token_ids in tokenized:
        if len(token_ids) > content_limit:
            head_size = max(1, int(content_limit * 0.7))
            tail_size = content_limit - head_size
            token_ids = token_ids[:head_size] + token_ids[-tail_size:]
        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
    return tokenizer.pad(
        rows,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
