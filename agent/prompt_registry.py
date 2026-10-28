# -*- coding: utf-8 -*-
"""不可变 Prompt 版本清单。

版本名、变更说明和内容哈希进入决策日志；旧日志因此不会在修改 Prompt 后失去可解释性。
"""
from __future__ import annotations

import hashlib
import os

from config import DEFAULT_PROMPT_VERSION


PROMPT_VERSIONS = {
    "manager-long-v1": {
        "description": "长期投资：三年论证、252日阶段观察、季度复核；区分核心补证与估值等待。",
        "suffix": "",
    },
    "manager-v1": {
        "description": "第二阶段基线：强制三专家顺序、风险终审与完整候选池预测。",
        "suffix": "",
    },
    "manager-v2": {
        "description": "第三阶段候选：要求证据分级并把置信度解释为可校准概率。",
        "suffix": (
            "\n\n置信度与证据规则（v2）：confidence 是方向命中的主观概率，必须在 0 到 1；"
            "只有至少两个相互独立的数据类别（例如价格行为与公告/基本面）一致时才能高于 0.70；"
            "新闻缺失、来源单一或证据冲突时不得高于 0.60。理由必须指出主要反证与失效条件。"
        ),
    },
}


def resolve_prompt_version(value: str | None = None) -> str:
    version = (value or os.environ.get("ASTOCK_PROMPT_VERSION", DEFAULT_PROMPT_VERSION)).strip()
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"未知 ASTOCK_PROMPT_VERSION={version!r}，可选:{', '.join(PROMPT_VERSIONS)}")
    return version


def render_versioned_prompt(base_prompt: str, version: str | None = None) -> tuple[str, dict]:
    resolved = resolve_prompt_version(version)
    rendered = base_prompt + PROMPT_VERSIONS[resolved]["suffix"]
    return rendered, prompt_manifest(rendered, resolved)


def prompt_manifest(rendered_prompt: str, version: str | None = None,
                    template_prompt: str | None = None) -> dict:
    resolved = resolve_prompt_version(version)
    manifest = {
        "version": resolved,
        "description": PROMPT_VERSIONS[resolved]["description"],
        "sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
    }
    if template_prompt is not None:
        manifest["template_sha256"] = hashlib.sha256(template_prompt.encode("utf-8")).hexdigest()
    return manifest
