#!/bin/bash
# scripts/run_all_tests.sh — Mosaic 全量测试运行脚本
set -e

echo "================================"
echo "  Mosaic 全量测试"
echo "================================"

echo ""
echo "--- Phase 1: 框架核心 + 文本域 ---"
pytest tests/phase1/ -v --tb=short

echo ""
echo "--- Phase 2: 图像域 ---"
pytest tests/phase2/ -v --tb=short

echo ""
echo "--- Phase 3: 音频域 + 字幕域 ---"
pytest tests/phase3/ -v --tb=short

echo ""
echo "--- Phase 4: 视频域 + 导出域 ---"
pytest tests/phase4/ -v --tb=short

echo ""
echo "--- Phase 5: RAG 域 ---"
pytest tests/phase5/ -v --tb=short

echo ""
echo "--- Phase 6: 一致性域 ---"
pytest tests/phase6/ -v --tb=short

echo ""
echo "--- Phase 7: 数字人域 ---"
pytest tests/phase7/ -v --tb=short

echo ""
echo "--- TTS 扩展: 全部 TTS 后端 ---"
pytest tests/tts/ -v --tb=short

echo ""
echo "--- Final: 最终验收 ---"
pytest tests/final/ -v --tb=short

echo ""
echo "================================"
echo "  测试完成"
echo "================================"