# 中文文本纠错API服务

基于深度学习的中文文本错误检测与纠正服务，支持GPT和MacBERT两种纠错模型。

## 功能特性

- 🚀 **高性能API服务**：基于FastAPI构建，支持异步处理
- 🧠 **多模型支持**：集成GPT和MacBERT纠错模型
- 📝 **单文本纠错**：支持单个文本的快速纠错
- 📚 **批量处理**：支持批量文本纠错，提高处理效率
- 🔍 **错误检测**：精确定位和标记文本中的错误位置
- 🐳 **容器化部署**：提供Docker和Docker Compose支持
- 📊 **健康监控**：内置健康检查和模型状态监控
- 📖 **API文档**：自动生成的交互式API文档

## 快速开始

### 方式一：本地开发运行

1. **安装依赖**
```bash
pdm install
```

2. **启动开发服务器**
```bash
pdm run serve
```

3. **启动生产服务器**
```bash
pdm run start
```

4. **访问服务**
- API服务：http://localhost:8000
- API文档：http://localhost:8000/docs
- ReDoc文档：http://localhost:8000/redoc

### 方式二：Docker容器运行

1. **构建并启动服务**
```bash
docker-compose up --build
```

2. **后台运行**
```bash
docker-compose up -d
```

3. **查看日志**
```bash
docker-compose logs -f
```

4. **停止服务**
```bash
docker-compose down
```

## API端点

### 基础端点

- `GET /` - 服务根路径
- `GET /health` - 健康检查
- `GET /models` - 获取可用模型列表

### 纠错端点

- `POST /correct` - 单个文本纠错
- `POST /correct/batch` - 批量文本纠错

## 使用示例

### 1. 健康检查

```bash
curl -X GET "http://localhost:8000/health"
```

### 2. 单个文本纠错

```bash
curl -X POST "http://localhost:8000/correct" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天新情很好",
    "model_type": "gpt"
  }'
```

### 3. 批量文本纠错

```bash
curl -X POST "http://localhost:8000/correct/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["今天新情很好", "这就是生或啊"],
    "model_type": "gpt"
  }'
```

### 4. Python客户端示例

```python
import requests

# 创建客户端
client = requests.Session()
base_url = "http://localhost:8000"

# 单个文本纠错
response = client.post(f"{base_url}/correct", json={
    "text": "今天新情很好",
    "model_type": "gpt"
})
result = response.json()
print(f"原文: {result['data']['source']}")
print(f"纠正: {result['data']['target']}")
```

运行完整示例：
```bash
python example_client.py
```

## 支持的模型

### GPT模型 (`gpt`)
- 模型：`shibing624/chinese-text-correction-1.5b`
- 特点：基于生成式预训练模型，支持批量处理
- 适用：通用文本纠错，语法和用词错误

### MacBERT模型 (`macbert`)
- 模型：`shibing624/macbert4csc-base-chinese`
- 特点：专门针对中文拼写检查优化
- 适用：拼写错误检测和纠正

## 响应格式

### 单个文本纠错响应

```json
{
  "success": true,
  "data": {
    "source": "今天新情很好",
    "target": "今天心情很好", 
    "errors": [
      {
        "original": "新",
        "corrected": "心",
        "position": 2
      }
    ]
  },
  "message": "纠错完成",
  "processing_time": 0.156
}
```

### 批量纠错响应

```json
{
  "success": true,
  "data": [
    {
      "source": "今天新情很好",
      "target": "今天心情很好",
      "errors": [...]
    }
  ],
  "message": "批量纠错完成",
  "processing_time": 0.312,
  "total_count": 1
}
```

## 配置说明

### 环境变量

- `PYTORCH_ENABLE_MPS_FALLBACK=1` - 启用MPS回退
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` - MPS内存管理

### 服务配置

- 默认端口：8000
- 最大请求文本长度：10,000字符
- 批量处理最大数量：100个文本

## 部署建议

### 生产环境

1. **使用反向代理**（如Nginx）
2. **配置SSL证书**
3. **设置合理的资源限制**
4. **启用日志记录和监控**
5. **定期备份模型文件**

### 性能优化

1. **模型缓存**：首次启动会下载模型，建议挂载缓存目录
2. **并发处理**：可通过增加worker数量提高并发能力
3. **GPU加速**：在支持CUDA的环境中可启用GPU加速

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接，确保能下载HuggingFace模型
   - 检查磁盘空间是否充足

2. **内存不足**
   - 减少batch_size或使用CPU模式
   - 增加系统内存

3. **服务启动慢**
   - 首次启动需要下载模型，请耐心等待
   - 后续启动会使用缓存，速度较快

### 日志查看

```bash
# Docker环境
docker-compose logs -f pycorrector-api

# 本地环境
pdm run serve  # 开发模式，显示详细日志
```

## 开发指南

### 项目结构

```
pycorrector/
├── src/pycorrector/
│   ├── __init__.py      # 原始测试脚本
│   ├── api.py           # FastAPI服务主文件
│   ├── models.py        # Pydantic数据模型
│   ├── demo.py          # 模型测试示例
│   ├── custom.txt       # 自定义词表
│   └── test.txt         # 测试文本
├── Dockerfile           # Docker构建文件
├── docker-compose.yml   # Docker Compose配置
├── example_client.py    # 客户端使用示例
└── pyproject.toml       # 项目配置和依赖
```

### 添加新模型

1. 在`api.py`的`load_models()`函数中添加新模型
2. 更新`models.py`中的模型类型验证
3. 在`process_correction_result()`中处理新模型的响应格式

## 许可证

MIT License
