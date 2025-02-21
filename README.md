# qniverse
Qniverse（Quantitative Trading’s Universe）是一个基于前沿人工智能和机器学习技术的开源量化交易算法项目。项目涵盖从传统方法到深度神经网络等多种技术，致力于构建一个统一框架，为不同类型的量化算法提供公平、透明、开放的回测与评估系统。同时，项目通过适配现有的量化交易开源库，方便广大开发者进行灵活使用和高效测试。


## 信号预测类开源算法
目前系统实现了以下9个信号预测类算法：

* XGBoost: A Scalable Tree Boosting System （XGBoost） [24]：XGBoost是基于GBDT的改进方法，被广泛用应用于多种机器学习任务。通过缓存优化和分片技术等改进，XGBoost能够高效处理大规模数据集。


* LightGBM: a highly efficient gradient boosting decision tree （GBDT） [25]：LightGBM基于GBDT并提出梯度一侧采样（GOSS）和互斥特征捆绑（EFB）两项关键技术来提升方法在大数据场景下的效率。GOSS通过筛除梯度较小的数据，仅使用梯度较大的样本计算信息增益；EFB通过将互斥特征捆绑在一起，减少特征数量，优化计算开销。实验表明，LightGBM相比传统GBDT在训练速度上提升超过20倍，同时几乎不损失精度。

* WFTNet: Exploiting Global and Local Periodicity in Long-term Time Series Forecasting （WFTNet）[99]：WFTNet提出了一种结合傅里叶变换和小波变换的长时间序列预测网络，用于同时提取信号的全局和局部频率信息。傅里叶变换捕捉全局周期模式，而小波变换则捕捉局部频率特征。此外，WFTNet引入了周期加权系数（PWC），自适应地平衡全局和局部频率模式的重要性。

* Periodicity Decoupling Framework for Long-term Series Forecasting （PDF）[100]：该算法提出了一种新的周期解耦框架（Periodicity Decoupling Framework, PDF），通过将一维时间序列解耦为二维时间变化模式，改进长时间序列预测性能。核心组件包括多周期解耦模块（MDB）、双变化建模模块（DVMB）和变化聚合模块（VAB），分别用于提取周期性特征、建模短期与长期变化，继而生成最终预测。

* TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis  (TimesNet)[101]：TimesNet通过将一维时间序列转化为二维张量，并基于多周期解耦，将时间序列的周期内、周期间变化嵌入二维张量，并利用参数高效的TimesBlock提取复杂的二维变化。TimesNet在预测、插补、分类和异常检测等五类任务中均表现出色。

* A Time Series is Worth 64 Words: Long-term Forecasting with Transformers （PatchTST）[136]：PatchTST为基于Transformer的多变量时间序列预测模型，核心贡献包括子序列分割和通道独立设计。子序列分割保留了局部语义信息，显著降低了计算和内存需求；通道独立聚焦于单变量时间序列信息提取，减少了过拟合问题并提高了模型的泛化能力。

* TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting （TimeMixer）[138]：TimeMixer为基于MLP的架构，通过Past-Decomposable-Mixing（PDM）和Future-Multipredictor-Mixing（FMM）模块，利用多尺度混合的方式处理复杂变化。PDM用于将季节性与趋势性成分进行解耦，FMM则通过集成多个预测器以增强模型的多尺度预测能力。

* TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting （TimeBridge）[168]：TimeBridge提出了一种新框架，通过集成注意力和协整注意力两种技术，用于解决时序预测中非平稳性的挑战。框架将输入时间序列划分为小片段，使用集成注意力机制捕捉各变量内的稳定依赖，同时通过协整注意力模块对变量间的长期协整关系进行建模。

* SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting （SegRNN）[169]：SegRNN引入了分段迭代和并行多步预测（PMF）两种策略，显著减少了RNN在长程时序预测任务中的迭代次数，使其在预测精度、运行速度和内存占用率方面均优于Transformer模型。

## 投资组合优化类开源算法
目前实现了以下12个投资组合优化类算法：

* PAMR: Passive aggressive mean reversion strategy for portfolio selection （PAMR）[114]：PAMR通过结合价格的均值回归特性与积极学习技术，实现投资组合中收益与波动风险的平衡。此外，文献还提供了多种变体算法，用于考虑交易手续费等因素。

* On-line portfolio selection with moving average reversion（OLMAR）[115, 116]： OLMAR是一种在线投资组合选择策略，通过多周期均值回归（MAR）克服单周期假设的局限性。实验表明，OLMAR不仅效果优异，还具有极高的运行速度，在诸多真实金融数据集上获得理想收益。

* Transaction cost optimization for online portfolio selection （TCO）[170]： TCO框架旨在改进在考虑手续费等交易成本条件下的在线投资组合选择策略。该框架在最大化期望回报的同时，尽量减少换手率和交易频率，与行业中的比例组合再平衡（PPR）原则一致。

* Weighted Moving Average Passive Aggressive Algorithm for Online Portfolio Selection （WMAMR）[171]：WMAMR是一种被动积极（Passive Aggressive）算法，通过引入移动平均损失函数实现多周期均值回归，有效提升了在线投资组合优化的表现。

* CORN: Correlation-driven nonparametric learning approach for portfolio selection （CORN）[172]：CORN通过非参数学习方法，利用股票价格间的统计关系来辅助股票交易决策。实验结果表明，CORN在各类真实股票市场上表现优异，能够显著超越市场指数。

* Nonparametric kernel-based sequential investment strategies（BNN）[173]：本文提出了一种序列投资策略，旨在通过核方法和最近邻学习技术实现资本的最优增长率。实证结果显示，在NYSE股票数据和外汇数据上，该策略均获得优异表现。

* Algorithms for portfolio management based on the Newton method（ONS）[174]：本文结合“最优对数遗憾界”（Optimal Log Regret Bound）计算技术，并利用牛顿法进行离线投资组合优化。实验表明，该方法在计算速度和效果上均表现优异。

* Can We Learn to Beat the Best Stock （Anticor）[175]：该方法基于股票间的统计关系，实现了一种稳健的投资组合优化策略，能够超越市场中表现最好的股票。

* On-Line Portfolio Selection Using Multiplicative Updates （EG）[176]：该方法基于乘法更新（Multiplicative Updates）规则，其资产增长率近似于理论最优的BCRP投资策略。此外，该算法实现简单高效，每个交易周期仅需常数级计算时间。

* Universal Portfolios（UP）[177]：该方法旨在构建通用的投资策略，能够主动适应市场变化，进而实现长期资产增长率的最大化。其核心思想是将资金分配到各类资产上，从而以信息论为基础系统地降低风险。

* A New Interpretation of Information Rate（Kelly）[178]：该准则以最大化资产长期增长率为目标，决定每次投入的资金量，适用于股票市场、赌博等不确定性较强的投资环境。

* Modern portfolio theory（MPT）[179]：现代投资组合理论（即均值-方差分析）通过凸优化方法综合地最大化期望收益、最小化方差，投资者根据这一技术，能够在可承担风险水平下实现最优资产配置。
