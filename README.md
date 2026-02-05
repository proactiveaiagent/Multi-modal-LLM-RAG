# 多模态主动式AI Agent系统（Proactive AI Agent with Multimodal RAG）

## 项目概述

本系统是一个基于RAG（检索增强生成）技术的多模态AI Agent，能够分析视频的视觉内容和音频信息，通过检索历史经验库提供主动式提醒和决策建议。https://github.com/proactiveaiagent/Multi-modal-LLM-RAG/blob/main/README.md

---

## 系统要求

- **Python**: 3.10
- **CUDA**: 12.1
- **GPU**: NVIDIA A800/A100
- **内存**: 32GB+
- **存储**: 1TB+

---

## 快速开始

### 第一步：克隆仓库
```bash
git clone https://github.com/proactiveaiagent/Multi-modal-LLM-RAG.git
cd Multi-modal-LLM-RAG
```

### 第二步：环境配置
```bash
conda create -n proactive_agent python=3.10
conda activate proactive_agent

pip install -r requirements.txt
```

或手动安装依赖：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.6 sentence-transformers==5.2.2 chromadb==1.4.0
pip install openai-whisper==20250625 opencv-python-headless==4.11.0.86
pip install pillow==10.4.0 numpy==1.26.0 pandas==2.3.3
```

---

```markdown
## 部署步骤

### 第一步：下载模型

#### 1. LLaVA-1.5-7B多模态模型

**方式一：HuggingFace下载**
```bash
mkdir -p models/llava-1.5-7b
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir models/llava-1.5-7b
```

**方式二：ModelScope魔塔下载**
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('lmms-lab/llava-v1.5-7b', cache_dir='models/llava-1.5-7b')"
```

#### 2. SentenceTransformer向量化模型

**方式一：HuggingFace下载**
```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir models/sentence-transformers
```

**方式二：ModelScope魔塔下载**
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('iic/nlp_corom_sentence-embedding_chinese-base', cache_dir='models/sentence-transformers')"
```

#### 3. Whisper语音识别模型

**方式一：自动下载（首次运行时）**
系统会在首次运行 `build_audio_memory.py` 时自动下载

**方式二：ModelScope魔塔下载**
```bash
python -c "from modelscope import snapshot_download; snapshot_download('iic/speech_whisper-base_asr_multilingual', cache_dir='models/whisper')"
```

**方式三：HuggingFace下载**
```bash
huggingface-cli download openai/whisper-base --local-dir models/whisper/whisper-base
```

### 第二步：准备测试视频

将测试视频按场景分类放入 `videos/` 目录

### 第三步：构建知识库

#### 初始化RAG知识库
```bash
python build_general_knowledge.py
```

**预期输出：**
```
Current records: 0
Adding 15 general scenarios...
Done! Total records: 41
```

#### 构建音频记忆库
```bash
python build_audio_memory.py
```

**预期输出：**
```
Processing 24 videos...
Extracting audio: 100%|████████| 24/24
Done! Total audio records: 24
```

### 第四步：运行测试
```bash
python proactive_agent_rag.py
```

**预期输出：**
```
============================================================
Proactive Agent with RAG Memory System
============================================================
Initializing multimodal memory...
Visual Memory: 41 experiences
Audio Memory: 24 transcripts
System Status: Ready

Processing 24 videos
[1/24] video1_neighbor_greeting.mp4
```

结果保存在 `output/` 目录

---

## 技术栈

- **视觉分析**: LLaVA-1.5-7B
- **音频识别**: Whisper Base
- **向量化**: SentenceTransformer (BERT)
- **向量数据库**: ChromaDB
- **深度学习框架**: PyTorch 2.7.0
- **硬件加速**: CUDA 12.1

---

## 项目结构
```
proactive_agent/
├── videos/                         
├── test_memory_db/                 
├── audio_memory_db/                
├── output/                         
├── proactive_agent_rag.py          
├── build_general_knowledge.py      
├── build_audio_memory.py           
├── requirements.txt                
└── README.md                       
```

---

## License

MIT License

## Acknowledgments

- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Whisper](https://github.com/openai/whisper)
- [ChromaDB](https://www.trychroma.com/)
- [SentenceTransformers](https://www.sbert.net/)
