# RepoRank

> 大多数 coding agent 把代码检索当成 grep。这个项目把它当成排序问题。

一个从零实现的研发提效 Agent。核心差异化在于把「代码上下文检索」按信息检索的方法论重做：稀疏检索 + 稠密检索 + 融合 + 重排 + 调用图先验，并提供完整的离线评测。

---

## 结果

> 待填。这张表是本项目的主结论，每完成一个模块补一行。
> 公开参照：SWE-bench Lite 文件级定位，BM25-Lucene 约 33.7%，Agentless 1.5 + GPT-4o 约 69.7%。

| 方案 | Recall@1 | Recall@5 | Recall@10 | MRR | NDCG@10 | 索引耗时 |
|---|---|---|---|---|---|---|
| BM25（朴素分词） | 0.200 | 0.440 | 0.540 | 0.600 | 0.312 | 33s |
| BM25 + 代码分词 | 0.300 | 0.620 | 0.740 | 0.800 | 0.452 | 70s |
| + 路径加权 | **0.340** | **0.640** | 0.740 | **0.820** | **0.489** | 69s |
| **+ 调参 (k1=1.2, ×10)** | **0.460** | **0.660** | 0.720 | 0.820 | **0.550** | 69s |
| AST 切分（公平对照） | 0.360 | 0.600 | 0.660 | 0.740 | 0.453 | 196s |
| Dense only | | | | | | |
| Hybrid (RRF) | | | | | | |
| + Cross-Encoder 重排 | | | | | | |
| + 调用图 PageRank | | | | | | |


w1-2 
### 基线结论（SWE-bench Lite, n=50）

- Recall@1 = 34.0%，与公开报告的 BM25-Lucene 约 33.7% 吻合，验证实现正确性
- **代码分词器**贡献 +10pt Recall@1，代价是索引耗时翻倍（33s → 70s）
- **路径加权**贡献 +4pt，耗时无变化，且仅改变排序不改变召回集合（Recall@10 前后均 0.740）
- **Recall@20 (0.820) 与 Recall@1 (0.340) 的 48pt 差距**表明多数失败是排序问题而非召回问题，重排层存在显著提升空间
- Recall@20 = Recall@30 = 0.820，剩余 18% 为硬失败，扩大检索窗口无效，需依赖语义检索或更细粒度切分

数据集全部为单文件修改（多文件比例 0%），此时 Recall@k 与 Hit@k 数学等价，
NDCG@k 退化为 MRR 的截断形式，故表中仅报告 Recall 与 MRR。评测代码保留全部
四项指标，以便后续迁移至 SWE-bench Verified（多文件比例约 14%）时使用。


### 核心发现（待验证）

检索召回率与端到端修复成功率**并非单调正相关**。SWE-bench 原论文已指出增大检索窗口会提升召回但因无关上下文干扰而降低端到端表现。本项目将量化这个拐点。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1. 生成离线 fixture，验证管道
python scripts/make_fixture.py
python scripts/run_eval.py --data fixtures/sample_instances.jsonl \
    --local-corpus fixtures/mini_repo --retriever bm25 --top-k 6

# 2. 消融对照：关掉代码分词器
python scripts/run_eval.py --data fixtures/sample_instances.jsonl \
    --local-corpus fixtures/mini_repo --retriever bm25-naive --top-k 6

# 3. 真实数据
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_data.py --out data/swebench_lite.jsonl
python scripts/run_eval.py --data data/swebench_lite.jsonl --retriever bm25 --limit 50
```

---

## 设计说明

### 为什么 ground truth 是免费的

SWE-bench 每条实例自带 gold patch（人类真实提交的修复）。从 unified diff 解析出被修改的文件路径，就是文件定位任务的标准答案。**不需要跑测试、不需要 Docker**，整个检索层的实验可以在笔记本上跑完。

### 为什么代码需要专门的分词器

`getUserName` 在通用分词器眼里是一个词，而 issue 里写的是 "get user name"，匹配不上。本项目按驼峰、下划线、数字边界拆分标识符，同时保留原始形式（精确匹配用）。fixture 上的对照实验：朴素分词 MRR 0.667，代码分词 MRR 1.000。

### 为什么路径要单独加权

文件路径是极强信号——issue 提到 auth 时 `src/auth/` 下的文件大概率相关。BM25 没有字段加权机制，这里用重复 token 的方式实现 field boosting，`path_boost` 是待扫的超参。

### 评测的健全性检查

`gold_not_in_corpus` 指标必须单独报出。如果 gold 文件被语料过滤规则误杀，那是语料构建的 bug，不是检索器的问题。不区分这两者会严重误判优化效果。

---

## 目录结构

```
reporank/
├── data/
│   ├── swebench.py     # 实例加载 + gold patch 解析
│   └── repo.py         # 仓库快照导出 + 语料构建
├── index/
│   └── tokenizer.py    # 代码感知分词器
├── retrieval/
│   ├── base.py         # Retriever 接口（所有方案的插槽）
│   └── bm25.py         # 稀疏检索基线
└── eval/
    ├── metrics.py      # Recall@k / Precision@k / MRR / NDCG@k
    └── runner.py       # 评测主循环 + 结果落盘
```

新增检索方案只需实现 `Retriever` 接口并在 `scripts/run_eval.py` 的注册表加一行，评测代码不用改。

---

## 路线图

- [x] W1-W2 BM25 基线 + 评测框架
- [ ] W3 AST 切分
- [ ] W4-W5 稠密检索 + RRF 融合
- [ ] W6 Cross-Encoder 重排
- [ ] W7 调用图 PageRank 先验
- [ ] W8 Workflow 架构（定位 → 修复 → 验证）
- [ ] W9-W10 ReAct Agent 架构
- [ ] W11 三层记忆机制
- [ ] W12 端到端评测 + 双架构对比
- [ ] W13 可靠性层 + 可观测层
- [ ] W14 LangChain 对照实现 + 服务化

## License

Apache-2.0
