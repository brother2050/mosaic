# $node_name

$description

## 基本信息

| 项目 | 值 |
|------|------|
| 域 (Domain) | `$domain` |
| 版本 | `0.1.0` |
| 作者 | `$author` |
| 许可证 | `$license` |
| 模型 | `$model_name` |

## 数据契约

- **输入类型:** $input_types
- **输出类型:** $output_types

## 使用示例

```python
from $node_name_snake import $node_name
from mosaic.core.types import MosaicData

node = $node_name()
node.load()
result = node(MosaicData(text="示例输入"))
print(result)
node.unload()
```

## 开发说明

本节点由 `mosaic create-node` 自动生成。请根据业务需求实现以下方法：

- `load()` — 加载模型
- `unload()` — 释放资源
- `run()` — 执行核心逻辑

> 标记为 **TODO** 的部分需要开发者补充实现。
