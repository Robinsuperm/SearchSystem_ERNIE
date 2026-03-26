# 基于 ERNIE 的两阶段语义搜索排序系统

## 项目简介
本项目实现了一个基于 ERNIE 的两阶段语义搜索排序系统，包括召回阶段和排序阶段。

- **召回阶段**：使用 HNSW 构建 ANN 索引库，实现高效向量检索。
- **排序阶段**：采用 Pairwise Matching Ranker，使用 margin ranking loss 进行精细排序。
