# 新能源智能体应用

基于FastAPI和LangChain构建的智能体应用，支持新能源数据的处理、检索和内容生成。

## 项目概述

本项目是一个专注于新能源领域的智能体应用，通过结合向量数据库、RAG技术和大型语言模型，提供高质量的信息检索与内容生成服务。应用支持多模态输入（文本和图像），并能生成专业、准确的新能源领域内容。

## 功能特点

- **多模态输入支持**：处理文本文档和图像输入
- **统一知识库**：将不同格式的数据统一存储和检索
- **高效检索**：通过向量存储和多路召回策略提高检索质量
- **内容生成**：基于RAG技术生成高质量新能源领域内容
- **指标验证**：评估生成内容的专业术语准确性、相关性和时效性
- **多格式导出**：支持PDF、HTML等多种格式导出，包含自定义样式

## 技术架构

- **后端框架**：FastAPI
- **RAG引擎**：LangChain + OpenAI模型
- **向量数据库**：Chroma DB
- **文档处理**：支持PDF、文本、CSV、Excel等格式
- **图像处理**：支持常见图像格式，包含OCR文本提取

## 开发计划

### 第一周：智能体应用生成
1. **知识库模型选择与实现**
   - 实现支持图片和文字的输入输出功能
   - 配置向量存储和模型接口

2. **基础数据库构建**
   - 统一数据格式
   - 数据上传与存储
   - 基础命中测试（图片、文档）

3. **知识库优化配置**
   - 向量参数优化
   - 标签系统实现
   - 域值调整以提升召回率
   - 应用打包与发布

### 第二周：RAG增强与生成优化
1. **基础生成能力测试**
   - 设定生成目标
   - 文本生成测试
   - 图文混合生成测试

2. **应用发布与导出功能**
   - 程序接口实现
   - PDF合成与导出功能

3. **RAG增强配置**
   - 多路召回策略实现
   - 重排序机制开发
   - 模型参数优化

4. **Prompt优化**
   - 设计专业化prompt模板
   - 优化生成引导策略

### 第三周：质量优化与导出完善
1. **验证指标实现**
   - 专业名词准确性评估
   - A/B测试机制（增强/弱化）
   - 时效性评估
   - 根据目标调整生成参数

2. **文档导出增强**
   - 表格样式支持
   - CSS样式定制
   - 计算功能实现

## 安装与配置

### 环境要求
- Python 3.10+
- 支持向量计算的环境

### 安装步骤

1. 克隆项目仓库
```bash
git clone https://github.com/your-username/new-energy-agent.git
cd new-energy-agent
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
# 创建.env文件
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 运行应用

启动FastAPI服务器:
```bash
uvicorn main:app --reload
```

应用将在 http://localhost:8000 运行，API文档可通过 http://localhost:8000/docs 访问。

## API接口

### 文档管理
- `POST /documents/upload`: 上传文档或图像到知识库
- `POST /knowledge-base/test`: 测试知识库检索能力

### RAG配置
- `POST /rag/configure`: 配置RAG引擎的不同策略

### 内容生成
- `POST /generate`: 基于查询使用RAG生成内容
- `POST /export`: 将生成的内容导出为不同格式

## 使用示例

### 文档上传
```python
import requests

url = "http://localhost:8000/documents/upload"
files = {"file": open("example.pdf", "rb")}
data = {"document_type": "pdf", "tags": "solar,energy"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 内容生成
```python
import requests
import json

url = "http://localhost:8000/generate"
payload = {
    "query": "太阳能光伏板的最新效率突破",
    "document_types": ["text", "image"],
    "tags": ["solar", "efficiency"],
    "generation_config": {
        "max_tokens": 1000,
        "temperature": 0.7,
        "include_sources": True
    }
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

## 自定义扩展

### 添加新的检索策略
扩展 `app/services/rag_engine.py` 中的 `RagEngine` 类，添加自定义策略实现。

### 自定义导出模板
在 `app/static/templates` 目录中添加HTML模板，自定义文档导出样式。

## 贡献指南

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件
