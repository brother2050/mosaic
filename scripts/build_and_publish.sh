#!/bin/bash
# scripts/build_and_publish.sh — Mosaic 打包发布脚本
set -e

echo "================================"
echo "  Mosaic 打包发布"
echo "================================"

# 清理旧的构建产物
echo ""
echo "--- 清理旧构建产物 ---"
rm -rf dist/ build/ *.egg-info/

# 构建 sdist 和 wheel
echo ""
echo "--- 构建 sdist 和 wheel ---"
python3 -m build

# 验证包
echo ""
echo "--- twine check 验证包 ---"
twine check dist/*

echo ""
echo "================================"
echo "  构建完成"
echo "================================"
echo ""
echo "# 发布到 TestPyPI（测试用）:"
echo "#   twine upload --repository testpypi dist/*"
echo ""
echo "# 发布到 PyPI（正式发布）:"
echo "#   twine upload dist/*"
echo ""
echo "# 版本标签:"
echo "#   git tag v0.1.0"
echo "#   git push origin v0.1.0"