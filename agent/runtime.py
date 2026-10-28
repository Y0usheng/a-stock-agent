"""统一模型设置与跨角色调用预算（不记录密钥）。"""
from __future__ import annotations

import os
from agents import ModelSettings, RunHooks
from openai.types.shared import Reasoning


def positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")
    return value


def model_settings(*, parallel: bool = False) -> ModelSettings:
    effort = os.environ.get("ASTOCK_REASONING_EFFORT", "medium").strip()
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("GPT 6.1 Sol reasoning effort 必须为 low/medium/high/xhigh/max")
    return ModelSettings(
        reasoning=Reasoning(effort=effort),
        max_tokens=positive_env("ASTOCK_MAX_OUTPUT_TOKENS", 8192),
        parallel_tool_calls=parallel, store=False,
    )


class ModelBudget(RunHooks):
    """所有专家共用计数器；达到限额后阻止后续调用，已在途调用仍会计费。"""
    def __init__(self):
        self.request_limit = positive_env("ASTOCK_MAX_MODEL_REQUESTS", 40)
        self.token_limit = positive_env("ASTOCK_MAX_TOTAL_TOKENS", 160_000)
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        if self.requests >= self.request_limit:
            raise RuntimeError("模型调用次数达到上限，停止分析")
        if self.input_tokens + self.output_tokens >= self.token_limit:
            raise RuntimeError("已完成模型调用的 token 达到上限，停止后续分析")
        self.requests += 1

    async def on_llm_end(self, context, agent, response):
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

    def report(self) -> dict:
        return {"request_limit": self.request_limit, "token_limit": self.token_limit,
                "requests_started": self.requests,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "max_output_tokens": model_settings().max_tokens,
                "reasoning_effort": model_settings().reasoning.effort}
