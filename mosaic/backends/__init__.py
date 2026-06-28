# mosaic/backends/__init__.py
"""Mosaic 后端适配层包。

为不同的推理后端（HuggingFace / vLLM / Ollama / API 服务）提供
统一接口适配，使节点可以透明地切换底层推理引擎。
"""
