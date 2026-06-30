# TEXT 域 API 文档

本目录包含 `mosaic.nodes.text` 下所有节点的自动生成 API 文档。

## 节点列表

| 节点 | 说明 | 流式 |
|------|------|:----:|
| `TextGenerator` | 文本生成，支持任意 prompt，逐 token 流式输出 | ✓ |
| `Chat` | 多轮对话，支持 messages 格式，逐 token 流式输出 | ✓ |
| `TextRewriter` | 文本重写，多风格改写 |
| `Translator` | 多语言翻译，支持中/英/日/韩/法/德/西等 |
| `TextSummarizer` | 文本摘要，支持最大长度控制 |
| `TextClassifier` | 文本分类，支持自定义标签集 |