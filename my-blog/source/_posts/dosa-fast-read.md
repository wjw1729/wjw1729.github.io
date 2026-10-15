---
title: Dosa Fast Read
category: llm
slug: dosa-fast-read
date: 2026-10-16 01:09:00
tags:
  - Sparse Attention
  - Fast Paper Read
---

[https://arxiv.org/pdf/2610.13990](https://arxiv.org/pdf/2610.13990)

# Dosa 算法

自从工作以后，个人就很少有时间静下心来看这些论文。最近由于有一些空闲，而且还是想把这些内容捡起来，所以开了这个 Fast Paper Read 专题，来提升自己对大模型的理解。本次主要涉及到的知识点是 Sparse Attention。

小红书团队针对 Sparse Attention 提出了一种新的剪枝方法，来对当前的性能进行优化 ($O(n^2)$)。

# 问题定义

![](https://cdn.nlark.com/yuque/0/2026/png/71899354/1792080064790-dead03a6-5aa7-45f2-8a89-91be121c6b06.png)

![](https://cdn.nlark.com/yuque/0/2026/png/71899354/1792080123057-47dad6f7-ca49-4ffd-909e-82ba9c58f648.png)

我们需要对于每个 token，通过上述公式计算其相关性。
针对相关性的计算，小红书的研究人员提出一种新的剪枝方法，大幅降低了计算的复杂度。其思想大概是：初始化一个 seed（候选集合），用这个候选集合的 index score 来确定一个得分上下界 (lower, upper)，用一个数学缩放上界提前对整个搜索空间进行剪枝来降低复杂度。整个算法的思路大致如下：

```
Seed 完整打分 -> τ (Top-K 门槛下界)
    ↓
对每个 key：少算几头 n -> 得到上界 U_i
    ↓
  U_i < τ ?-是-> 丢弃 (无需完整打分)
    否
    ↓
完整 H 个 head 打分 -> 真实 s_i
    ↓
s_i >= τ 才写回 -> 与 seed 一起做最终 Top-K
```

# 核心算法

## 1. Seed 初始化与保留头的筛选

### Seed 初始化

选取互不相同的可见 key 组成种子集 ![](https://cdn.nlark.com/yuque/__latex/1a7307ad3e161f2a76da0690907daf89.svg)，且 ![](https://cdn.nlark.com/yuque/__latex/24b60f3f3fcb0a133dcae89f6d4c08d4.svg)，令 ![](https://cdn.nlark.com/yuque/__latex/0252d751303e298c938b248f176c71b7.svg) 为其第 ![](https://cdn.nlark.com/yuque/__latex/38a3f4d664b7a723d138f9d57be0c783.svg) 大的完整分数。该子集提供 ![](https://cdn.nlark.com/yuque/__latex/38a3f4d664b7a723d138f9d57be0c783.svg) 个“见证”：

![](https://cdn.nlark.com/yuque/__latex/076b0c0263f589e4f34a8f178acbcfc2.svg)

因此 ![](https://cdn.nlark.com/yuque/__latex/3dd67ff19126696cdc6727956c722d13.svg) 是真实 Top-![](https://cdn.nlark.com/yuque/__latex/38a3f4d664b7a723d138f9d57be0c783.svg) 阈值的下界，![](https://cdn.nlark.com/yuque/__latex/202a4da434d9cdf3093143fe0622cdcc.svg) 是 token 分数的上界：二者严格分离即可证明该 token 可被安全剔除。种子质量差会削弱过滤；额外的精确分数只会抬高阈值下界。

### 保留头的选取方法

从所有头 $H$ 中保留 $r$ 个头，设 ![](https://cdn.nlark.com/yuque/__latex/dd1caa3f2e1582dab2cf9bfdb21b7556.svg) 为保留头集合，满足 ![](https://cdn.nlark.com/yuque/__latex/f7d37705ca6584993dc1a49988a51ccf.svg)。定义 ![](https://cdn.nlark.com/yuque/__latex/a8db057e8551e1b8388d3d3d6963c058.svg)、![](https://cdn.nlark.com/yuque/__latex/3337dc71ac762633073fe513ae7d0124.svg)。

- 若 ![](https://cdn.nlark.com/yuque/__latex/591627047897cf55c3829ecbbe955f1a.svg)：保留全部正权重头，剩余位置用负权重头填补，优先级为 ![](https://cdn.nlark.com/yuque/__latex/8f74c9ed9d61aa9866abbcc6bce19658.svg)，其中 ![](https://cdn.nlark.com/yuque/__latex/0432f76dbe6461f34f024cb0a56a9d61.svg) 是该头在采样种子 key 上的激活频率。这优先保留“强且激活难预测”的头。
- 否则：按 ![](https://cdn.nlark.com/yuque/__latex/c627b59331e4feccf08caf0f03a14347.svg) 保留最强的 ![](https://cdn.nlark.com/yuque/__latex/72cb3a229067770aeb6caa625a65a1a1.svg) 个正权重头。
