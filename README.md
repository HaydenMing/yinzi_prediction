# 双物料五框架CUDA实验
仓库：https://github.com/HaydenMing/yinzi_prediction
本地目录：F:\暑期项目\GitHub同步\yinzi_prediction
服务器目录：/media/data2/xiawk/work2

## 内容
src/：独立原文编码器、五融合主体、训练、选优、结果导出。
data/raw/：两物料原新闻、原因子及原价格Excel。
data/reference/：固定样本、切分、价格标准化数组及清理记录。原manifest仅审计，其Windows路径不用于服务器寻址。
data/input_manifest.json：相对路径及SHA256。
数据包约20.4MB，不含API密钥或账号配置。原新闻正文和因子证据完整保留，模型因子字段为名称与解释。
预测下一报价日价格，另导出价差、百分比、派生方向。不是给因子预测涨跌强度。
设计见src/README.md及src/预测目标与因子说明.md；服务器命令以本文为准。

## 1. 服务器首次拉取
~~~bash
mkdir -p /media/data2/xiawk
git clone https://github.com/HaydenMing/yinzi_prediction.git /media/data2/xiawk/work2
cd /media/data2/xiawk/work2
~~~
已有此仓库时：
~~~bash
cd /media/data2/xiawk/work2
git pull --ff-only origin main
~~~
私有仓库自行完成GitHub的SSH或HTTPS认证，不把token放进代码或聊天。

## 2. CUDA环境
先看GPU与驱动：
~~~bash
nvidia-smi
~~~
下列安装使用CUDA12.4版PyTorch，需驱动支持该版本。已有合适CUDA版torch环境也可使用，环境写入结果。
~~~bash
conda create -n yinzi-gpu python=3.11 -y
conda activate yinzi-gpu
cd /media/data2/xiawk/work2
python -m pip install -r requirements-gpu.txt
~~~
可自行确认GPU4：
~~~bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
~~~
GPU4是物理编号4；只暴露这一张卡之后，程序内使用cuda:0。

## 3. GPU4完整训练
前台：
~~~bash
conda activate yinzi-gpu
cd /media/data2/xiawk/work2
bash scripts/run_gpu4.sh
~~~
或后台（与前台二选一）：
~~~bash
conda activate yinzi-gpu
cd /media/data2/xiawk/work2
mkdir -p logs
nohup bash scripts/run_gpu4.sh > logs/gpu4.log 2>&1 &
echo $!
tail -f logs/gpu4.log
~~~
480次候选训练=两物料×五框架×16方案×三种子。只用验证MAE选优，再测试选中的方案。
输出outputs/gpu4_formal。中断后用同一命令续跑，已完成单次复用，未完成单次重新训练。
固定随机种子，不默认开启强制确定性算法；GPU仍可能有细微非确定性。严格模式可直接调用src/run.py加--deterministic，CUDA不支持的确定性算子会报错。
模型和输入均移动到CUDA。读取和文字分块仍由CPU准备，无API或模型下载。
同一输出只接受同一代码、参数、环境。CPU与GPU结果使用不同目录。

## 4. 查看结果
outputs/gpu4_formal/所有实验结果.csv
outputs/gpu4_formal/导师汇报用结果.md
outputs/gpu4_formal/先后融合配对比较.csv
选中模型子目录：测试逐日报价预测.csv、features.npz、best.pt。
默认Git回传CSV、JSON、Markdown、图片和逐日预测；权重best.pt、特征features.npz留服务器，不通过普通Git回传。准备数据副本也不重复回传。需要这些大文件时另行传输。
原始输入data/下的dataset.npz已纳入Git。

## 5. 服务器推回结果
第一次提交前设置自己的Git身份：
~~~bash
git config user.name "HaydenMing"
git config user.email "162002191+HaydenMing@users.noreply.github.com"
~~~
完成训练后：
~~~bash
cd /media/data2/xiawk/work2
bash scripts/push_results.sh
~~~
脚本暂存outputs、提交、拉取更新后推送；发生冲突时先处理，不强推。

## 6. 本地下载（PowerShell）
~~~powershell
git -C 'F:\暑期项目\GitHub同步\yinzi_prediction' pull --ff-only origin main
~~~
结果位于 F:\暑期项目\GitHub同步\yinzi_prediction\outputs\gpu4_formal 。
只阅读结果即可；本地CPU不要续训服务器同一输出目录。

## 范围
本次没有启动训练、没有运行检查脚本、没有登录thu-vip。五框架是既有改图适配，early是改版，不是五篇原论文完整复现；历史测试区间被查看过，不是新盲测。
CUDA安装来源：https://docs.pytorch.org/get-started/previous-versions/
