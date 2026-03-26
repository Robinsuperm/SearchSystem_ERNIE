
# ERNIE-based Semantic Search Ranking System

**基于预训练语言模型的搜索排序系统**  
实现召回 + 排序两阶段架构，Recall@20 达到 0.813。

### 项目说明
本项目构建了一个完整的语义搜索系统，针对 30 万篇文献进行 query-title 匹配。  
采用两阶段架构：**ANN 召回 + ERNIE 精细排序**，有效解决大规模语义检索中的速度与精度问题。

### 核心贡献
- 召回阶段使用 ERNIE-1.0 将 title + 关键字编码为 256 维向量，采用In-batch Negative 对比学习、通过 ANN 构建索引，实现快速 Top-50 召回（Recall@20 = 0.813）  
- 排序阶段切换到 ERNIE-3.0，采用 Pairwise Matching，通过 Margin Ranking Loss 进行精细重排序  
- 完成数据处理、模型微调、ANN 索引构建及评估全流程

### 训练结果
- Recall@20 = 0.813（工业级优秀水平）  
