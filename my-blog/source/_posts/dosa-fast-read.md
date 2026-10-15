---
title: Dosa Fast Read
category: llm
slug: dosa-fast-read
date: 2026-10-16 01:09:00
mathjax: true
tags:
  - Sparse Attention
  - Fast Paper Read
---

[https://arxiv.org/pdf/2610.13990](https://arxiv.org/pdf/2610.13990)

# Dosa 算法
自从工作以后，个人就很少有时间静下心来看这些论文。最近由于有一些空闲，而且还是想把这些内容捡起来，所以开了这个Fast Paper Read专题，来提升自己对大模型的理解。本此主要涉及到的知识点是Sparse Attention。

小红书团队针对Sparse Attention提出了一种新的剪枝方法，来对当前的性能进行优化(O(n^2))。

# 问题定义
<!-- 这是一张图片，ocr 内容为：OUTPUT HIDDENUR [OOOOO .... . . . . . . . . . . . . . ........... . . . . . . . . . . . . . CELORIS MULTI-QUERY ATTENTION(CORE ATTENTION) MALALU {QTIIQIID TOP-K SELECTOR 100...00 CONCATENATE LIGHTNING FLEW;KBOO............. INDEXER CONCATENATE [QI.I了 APPLY ROPE PARTIALLY PARTIALLY APPLY ROPE APPLY ROPE APPLY ROPE [000 INPUT HIDDEN FIGURE 2 ATTENTION ARCHITECTURE OF DEEPSEEK-V3.2, WHERE DSA IS INSTANTIATED UNDER MLA. THE GREEN PART ILLUSTRATES HOW DSA SELECTS THE TOP-K KEY-VALUE ENTRIES ACCORDING TO TO THE INDEXER- -->
![](/images/dosa/attention-arch.png)



<!-- 这是一张图片，ocr 内容为：(ANG-K), RELU IT,S 三 T, -->
![](/images/dosa/index-score.png)

我们需要对于每个token，通过上述公式计算其相关性。

针对相关性的计算，小红书的研究人员提出一种新的剪枝方法，大幅降低了计算的复杂度。其思想大概是：初始化一个seed（候选集合），用这个后选集合的index score来确定一个得分上下界(lower, upper)，用一个数学缩放上界提前对整个搜索空间进行剪枝来降低复杂度。真个算法的思路大致如下：

```python
Seed 完整打分 -> τ (Top-K 门槛下界)
    ↓
对每个 key：少算几头n -> 得到上界 U_i
    ↓
  U_i < τ ?-是-> 丢弃 (无需完整打分)
    否
    ↓
完整H个head打分 ->真实 s_i
    ↓
s_i >= τ 才写回 -> 与seed一起做最终 Top-K
```

# 核心算法
## Seed初始化与保留头的筛选
Seed初始化

选取互不相同的可见 key 组成种子集 $ S \subseteq \Omega $，且 $ |S| \ge K $，令 $ \tau = \tau_K(S) $ 为其第 $ K $ 大的完整分数。该子集提供 $ K $ 个“见证”：



$ \tau_K(S) \le \tau_K(\Omega),\qquad

s_i \le U_i < \tau_K(S) \implies i \notin \mathrm{TopK}(\Omega) \tag{2} $

因此 $ \tau_K(S) $ 是真实 Top-$ K $ 阈值的下界，$ U_i $ 是 token 分数的上界：二者严格分离即可证明该 token 可被安全剔除。种子质量差会削弱过滤；额外的精确分数只会抬高阈值下界。



保留头的选取方法

从所有头H中保留r个头，设 $ R $ 为保留头集合，满足 $ 1 \le r < H $。定义 $ H_+ = \{h : w_h > 0\} $、$ H_- = \{h : w_h < 0\} $。

+ 若 $ |H_+| \le r $：保留全部正权重头，剩余位置用负权重头填补，优先级为 $ |w_h|\,\|q_h\|_2 \min(p_h, 1-p_h) $，其中 $ p_h $ 是该头在采样种子 key 上的激活频率。这优先保留“强且激活难预测”的头。
+ 否则：按 $ w_h \|q_h\|_2 $ 保留最强的 $ r $ 个正权重头。




## 省略正权重头的上界
对每个 $ h \in P $，选一个保留锚点与非负投影：



$ \begin{aligned}

j_h^{\star}

&= \arg\max_{j \in R}

\frac{\bigl[q_h^{\top} q_j\bigr]_+}{\|q_j\|_2}, \\[0.5em]

\beta_h

&= \frac{\bigl[q_h^{\top} q_{j_h^{\star}}\bigr]_+}{\|q_{j_h^{\star}}\|_2^{2}}, \\[0.5em]

d_h

&= q_h - \beta_h q_{j_h^{\star}}

\end{aligned} \tag{4} $

零范数锚点对应匹配分数与系数均为零。由 ReLU 次可加性、正齐次性与 Cauchy–Schwarz：



$ \bigl[q_h^{\top} k_i\bigr]_+

\le

\beta_h \bigl[q_{j_h^{\star}}^{\top} k_i\bigr]_+

+ \|d_h\|_2 \|k_i\|_2 \tag{5} $

投影部分可复用已算好的点积。非负投影最小化残差范数，在此类单锚点界中给出最小的一致范数裕量（见附录 B.1）。定义



$ \tilde{w}_j = w_j + \sum_{\substack{h \in P \\ j_h^{\star}=j}} w_h \beta_h,\qquad

E_P = \sum_{h \in P} w_h \|d_h\|_2 \tag{6} $

于是保留头与省略正权重项合起来可被



$ \sum_{j \in R} \tilde{w}_j \bigl[q_j^{\top} k_i\bigr]_+ + E_P \|k_i\|_2 $

上界。折叠权重仅属于筛选计划，不修改模型参数。若 $ P $ 为空，则 $ E_P = 0 $ 且 $ \tilde{w}_j = w_j $。



省略残差还共享同一个 key。令 $ u_h = w_h d_h $（$ h \in P $），联合裕量为



$ E_P^{\mathrm{joint}}

=

\frac{1}{2}

\left(

\left\|\sum_{h \in P} u_h\right\|_2

+

\sqrt{\sum_{h,\ell \in P}|u_h^{\top} u_\ell|}

\right)

\le E_P \tag{7} $

它用 $ E_P^{\mathrm{joint}} \|k_i\|_2 $ 上界 $ \sum_{h \in P}[u_h^{\top} k_i]_+ $，因而可替换 $ E_P $。改进来自残差几何，无需额外读 key。

## 省略负权重头的下界
负权重头从分数中做减法。构造上界时，我们对子集 $ A \subseteq N $ 做聚合，并忽略其余头：省略负贡献只会抬高上界。定义



$ a_A = \sum_{h \in A}|w_h| q_h,\qquad

L_N(k_i) = \bigl[a_A^{\top} k_i\bigr]_+

\le

\sum_{h \in N}|w_h|\bigl[q_h^{\top} k_i\bigr]_+ \tag{8} $

ReLU 次可加性保证 $ L_N $ 不超过省略负权重头真正减去的量。为何不把所有头都放进 $ A $？未激活的头经自身 ReLU 后贡献为零，但其负点积可能在聚合 ReLU 之前抵消已激活头的正点积。纳入这类头会削弱负贡献估计，从而放松分数上界。



我们用种子激活模式减少这种抵消。对采样种子 key，记录 $ M_i = \{h \in N : q_h^{\top} k_i > 0\} $，令 $ p_h $ 为头 $ h $ 的频率。取逐分量多数支持：



$ A = \{h \in N : p_h \ge 1/2\} $

这最小化归一化负松弛的可分上界；同时，上文保留负权重头的优先级会去掉该替代量中剩余最大的项（附录 B.2）。若新 key 的激活集恰好是 $ A $，则聚合精确；其他模式仍安全，但界可能更松。频率只指导效率，不影响正确性。

## 整合
综合上式，得到分数上界公式：



$ U_i

=

\sum_{j \in R}

\tilde{w}_j \bigl[q_j^{\top} k_i\bigr]_+

+ E_P \|k_i\|_2

- \bigl[a_A^{\top} k_i\bigr]_+,

\qquad

s_i \le U_i \tag{9} $

一个聚合方向可替代任意多个省略负权重头。其效用取决于激活一致性；正裕量则取决于方向相似性。





# Reference
+ [https://zhuanlan.zhihu.com/p/1962162900111172920](https://zhuanlan.zhihu.com/p/1962162900111172920)
