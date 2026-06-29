#!/bin/bash
# scripts/verify_release.sh — Mosaic 发布前验证脚本
set -e

echo "================================"
echo "  Mosaic 发布前验证"
echo "================================"

# 1. 检查 Python 版本
echo ""
echo "--- 1. Python 版本 ---"
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.10" || "$PYTHON_VERSION" == "3.11" || "$PYTHON_VERSION" == "3.12" ]]; then
    echo "  Python $PYTHON_VERSION OK"
else
    echo "  WARNING: Python $PYTHON_VERSION (expected 3.10/3.11/3.12)"
fi

# 2. 检查关键依赖
echo ""
echo "--- 2. 依赖检查 ---"
python3 -c "
import sys
deps = ['diffusers', 'transformers', 'torch', 'numpy', 'PIL', 'pytest', 'click']
for dep in deps:
    try:
        __import__(dep)
        print(f'  {dep}: OK')
    except ImportError:
        print(f'  {dep}: MISSING')
"

# 3. 运行全量测试
echo ""
echo "--- 3. 运行全量测试 ---"
bash scripts/run_all_tests.sh

# 4. 检查文档完整性
echo ""
echo "--- 4. 文档完整性 ---"
DOCS_DIR="docs"
EXPECTED_DOCS=(
    "getting-started.md"
    "architecture.md"
    "nodes-reference.md"
    "pipeline-guide.md"
    "tts-guide.md"
    "video-models.md"
    "plugin-development.md"
    "cli-reference.md"
)
DOC_COUNT=0
for doc in "${EXPECTED_DOCS[@]}"; do
    if [ -f "$DOCS_DIR/$doc" ]; then
        echo "  $DOCS_DIR/$doc: OK"
        ((DOC_COUNT++))
    else
        echo "  $DOCS_DIR/$doc: MISSING"
    fi
done
echo "  文档: $DOC_COUNT/${#EXPECTED_DOCS[@]}"

# 5. 检查示例文件完整性
echo ""
echo "--- 5. 示例文件完整性 ---"
EXAMPLES_DIR="examples"
EXAMPLE_COUNT=$(ls "$EXAMPLES_DIR"/*.py 2>/dev/null | wc -l)
echo "  示例文件数: $EXAMPLE_COUNT (expected 11)"

# 6. 检查关键文件
echo ""
echo "--- 6. 关键文件检查 ---"
for f in LICENSE README.md CHANGELOG.md MANIFEST.in setup.cfg; do
    if [ -f "$f" ]; then
        echo "  $f: OK"
    else
        echo "  $f: MISSING"
    fi
done

# 7. 运行 mosaic doctor
echo ""
echo "--- 7. mosaic doctor ---"
python3 -m mosaic.cli.main doctor

# 8. 版本检查
echo ""
echo "--- 8. 版本检查 ---"
python3 -m mosaic.cli.main version

echo ""
echo "================================"
echo "  验证完成"
echo "================================"