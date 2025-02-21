# Walkthrough

## 系统demo

- https://naiqili.github.io/project
- 核心要求输入：reporter/tmp

## 工作规划

近期：
- 将现有代码并入qniverse：WFTNet, PDF, TimesBridge, StockMamba...
- 在reporter/tmp输出图片结果
- 提供接口，在qlib系统中任意调用Time Series Library (TSLib)
- 提供接口，在qlib系统中任意调用Universal Portfolio
- 我们的方法纂写说明文档（毕业论文部分）

未来：
- 更好的类封装
- 整理代码（注释、readme、tutorial），最终开源
- 融入其它类型算法（RL）
- 统一benchmark、统一evaluation
- 综述论文

Misc:
- 收集之前工作的各类poster、公众号文章、GitHub等材料

## 悟空数据库

- db_test.ipynb
- 数据库转qlib: wk2qlib_cn.ipynb
- raw_data: .qlib/qlib_data

## Reporter Walkthrough

- reporter/model/gdbt_pred.py
- reporter/model/gdbt_fig.py

# GBDT Walkthrough
 
## GBDT Pipeline

- gbdt/bench.ipynb
- gbdt/reporter.ipynb

## 大规模测试调参

- 统一benchmark

    lilab/benchmark

- 测试脚本

    lilab/run_bench.sh

- Results

    BENCH_A/B/C.csv

    report_A/B/C.ipynb

## 可用投资决策辅助系统

- 限制资金，限制股池

    gdbt/realworld_test.ipynb

    gdbt/realworld_result/BENCH_A.csv

- 每日数据更新、数据预测、仓位构建维护全流程

    gdbt/realworld_position_maker.ipynb