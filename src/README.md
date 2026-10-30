# 五框架分模态编码实验
这是新实现目录。旧实验、原始新闻、原始因子和旧结果保持不变。本次只编写代码，没有执行Python、检查脚本或训练。
实验目标是下一报价日价格；不是预测因子，也不是由大模型生成价格。不调用API，不下载模型。

## 先看什么
1. 预测目标与因子说明.md：用于向导师解释输入、输出、公式。
2. model.py：三种模态怎样编码、何处拼接、何处融合。
3. encoders.py：真正处理原文的TextCNN，以及其他可替换编码器。
4. fusion.py：五框架融合主体和预测层。
5. data.py：原文、因子、训练词表、报价日窗口。
6. training.py：训练、早停、测试与导出。
7. 实验配置.json：实际跑的16个方案。
8. experiments.py：方案检查及验证选优范围。
9. run.py：组织完整实验。
10. reports.py：结果CSV、基线及汇报结果。
11. common.py：价格还原、指标及文件保存。
12. test_contracts.py：供用户自行运行的离线检查。

## 与旧版的实际区别
- 原版先把新闻/因子变成TF-IDF/SVD，再三个GRU。新版从新闻标题正文、因子名称解释的完整字符序列重新编码。
- 价格与原文不用强行选同一种编码器。三路独立模块，各自参数不共享。
- TextCNN沿字符序列卷积，卷积宽度3/4/5；不是对日期序列卷积后冒称TextCNN。
- 价格可选GRU或TCN；新闻、因子各选mean、TextCNN、BiGRU。
- mean是可训练字符Embedding均值基准。BiGRU在字符块内双向，不跨预测截止时间。
- late的新闻和因子原文编码之后，各有一个独立GRU处理报价日序列。它们不是原文编码器，配置和模型说明中明确列出。可在JSON增加news_temporal/factor_temporal选择gru或tcn。
- 原文按块保留全部内容。TextCNN采用4字重叠，并在最终投影之前跨块最大池化；mean分块加权均值；BiGRU分块编码后按字数加权，跨块没有完整的双向上下文。块长度是上下文范围，不是只保留前256字。
- 原文因子证据保留在因子输入目录.csv供核查。模型输入是因子名称和解释，证据不重复输入。
- 数据只使用历史缓存的价格标准化、目标尺度、样本日期和提取状态；不使用旧文字SVD特征。词表仅由训练输入窗口建立。

## 两种融合顺序
late，先提取再融合：
价格2维 → 价格GRU/TCN → price_features。
新闻原文 → 自己的Embedding及mean/TextCNN/BiGRU → 加5个状态 → 新闻GRU → news_features。
因子文字 → 自己的Embedding及mean/TextCNN/BiGRU → 加5个状态 → 因子GRU → factor_features。
三路特征一起进入指定框架的融合主体，再回归标准化价差。

early，先融合再提取：
价格2维 → 线性映射为一个Embedding向量；新闻和因子 → 字符Embedding；加模态类型和状态向量。
每个报价快照先把价格、状态、新闻、因子这些基础向量拼成joint_tokens。
共享mean/TextCNN/BiGRU提取joint_features，再用GRU/TCN处理报价日序列。
共享结果投影成三组mixed_views，进入五融合主体及预测层。
每个文字块都带上对应价格和状态前缀；全部文字参与。没有新闻时价格也照常参与。
这里改变的是基础表示对齐后的融合位置。early没有先用独立TextCNN处理新闻/因子。混合视图不再是独立模态，不能把其权重叫作价格/新闻/因子的因果权重。

五个融合主体继续沿用五张改图的适配，early是进一步修改的版本。①保留高斯瓶颈、队列和PN/PS代理监督，③不做NAS。不是五篇原论文完整复现。参数量不同，因此比较的是两种完整设计，不是严格保持容量不变的单因素实验。

## 实验矩阵
6个配对组合：
- 价格/共享时序GRU，文字mean。
- 价格/共享时序GRU，文字TextCNN。
- 价格/共享时序GRU，文字BiGRU。
- 价格/共享时序TCN，文字mean。
- 价格/共享时序TCN，文字TextCNN。
- 价格/共享时序TCN，文字BiGRU。
每个配对组合有early/late两个版本，共12方案。
额外4个late：固定价格GRU，分别只把新闻或因子从mean换成TextCNN/BiGRU。
共16方案 × 2物料 × 5框架 × 3种子 = 480次训练。不是所有可能编码器组合的穷举。

只按三个种子平均验证MAE选择：
- paired_late：6个配对late候选中选1个。
- paired_early：6个配对early候选中选1个。
- native_late：全部10个late候选中选1个，用于不同模态编码器比较。
三个用途各报告，重复选中的模型只测试一次。最多90次测试评估，不新增训练。
测试期已在之前实验查看过，属于诊断比较，不能宣称全新盲测。

## 运行环境
使用已有Python环境即可，新代码只需要NumPy与PyTorch。
requirements.txt只列最低依赖，不建议为了本次实验直接升级已能运行的环境；环境版本会写入manifest，续跑时需一致。
默认CPU、2线程、hidden=32、Embedding=32、batch=8、最多40轮、patience=8、种子11/22/33。

## 运行命令
下面均由用户自己执行。PowerShell设置两个变量，保留同一个输出路径可续跑：
~~~powershell
$code = 'F:\暑期项目\预测项目代码\五框架分模态编码\run.py'
$out = 'F:\暑期项目\预测结果\五框架分模态编码\formal_20261030'
~~~

只打印矩阵，不训练，不建立输出文件：
~~~powershell
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --plan-only
~~~

完整运行，包含两物料、五框架、480候选和选中模型测试：
~~~powershell
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --output $out --stage all
~~~

也可以先准备原文、词表、因子目录，审阅后再训练：
~~~powershell
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --output $out --stage prepare
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --output $out --stage train
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --output $out --stage evaluate
~~~
三个阶段使用同一参数、同一输出目录；evaluate只接受全部候选已经训练完成的目录，不补训练。
中断后重跑同一命令：已完成的模型跳过，尚未完成的一次训练从头开始。不是恢复优化器到中断步。
改动代码、配置、随机种子、批大小、环境或输入后必须改输出目录；不能混用结果。仅stage允许变化。

可选的小规模试运行，用另一个输出目录，不能当正式结果：
~~~powershell
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u $code --output 'F:\暑期项目\预测结果\五框架分模态编码\trial_20261030' --materials 磷酸铁 --models hierarchical --recipes late_gru_textcnn early_gru_textcnn --seeds 11 --epochs 1 --patience 1
~~~

可选离线检查（本次未执行）：
~~~powershell
& 'C:\Users\w89ha\venv\Scripts\python.exe' -X utf8 -u 'F:\暑期项目\预测项目代码\五框架分模态编码\test_contracts.py'
~~~

## 看哪些结果
- 所有实验结果.csv：默认完整运行后642行=2基线+160方案均值+480种子明细。未选中方案的测试指标留空。按行类型筛选，避免重复统计。
- 方案均值对比.csv：160个物料/框架/方案的验证和已有测试均值。不能按测试分数回头更换选择。
- 先后融合配对比较.csv：每个固定配对的验证差异和参数量。
- 验证选优.json：实际选择范围、候选数量及胜出方案。
- 导师汇报用结果.md：按验证选出的方案及其测试结果、持价基线。
- 全部选中方案逐日预测.csv：已测试模型逐日真实价格和预测价格。
- 选中方案种子平均预测.csv：多个种子预测价格均值；其误差不等于各种子MAE均值。
- data/物料/新闻输入目录.csv：窗口实际用到的新闻全文、时间、来源及提取状态。
- data/物料/因子输入目录.csv：实际用到的因子名称、解释、原文证据和新闻ID。不提供虚构的数值强度。
- data/物料/snapshots.json：完整文字、字符ID及五状态。
- data/物料/arrays.npz：价格输入p、标准化价差y、actual、baseline、dates、snapshot_ids。
- data/物料/vocabulary.json：只用训练文字拟合的字表。
- 物料/方案/框架_seed数字/验证逐日报价预测.csv。
- 物料/方案/框架_seed数字/测试逐日报价预测.csv：仅验证选中的方案生成。
- 同目录features.npz：仅测试选中模型导出。late有price_features、news_features、factor_features；early有joint_features和mixed_view_1/2/3；都有fusion_features、dates、normalized_prediction。
- 同目录model_config.json：完整编码器与时序模块配置、输出含义。
- best.pt：验证集最优权重；training.json：训练过程；validation_metrics.json：单次完成标记。
- manifest.json：代码、原始数据、环境、配置；prepared_inputs.json和ready.json校验准备数据。

## 汇报时应明确的范围
这是既有历史区间的下一报价预测评估，不是获取2026年最新行情后预测明天。
价格表计量单位尚未确认，价格和MAE沿用原文件尺度，不随意标元/吨。
因子格式和证据对应检查不能证明因子语义全部合格。沿用旧保守清理，完整人工语义审核仍未完成。
新闻和因子同源；字符Embedding从现有训练文本学习，未使用外部预训练语言模型。
不能仅凭程序完成就说预测有效；须对比持价基线、变动日表现、三种子波动，并按验证选择。
筛选方案后，只有双方共有pair_id进入paired_late/paired_early；不成对方案单列unpaired范围，不能用于同预算顺序比较。
