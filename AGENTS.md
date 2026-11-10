# RoboJEPA Worlds 工作规则

- 本仓库的目标是把**预训练 RoboJEPA 微调适配到已有工作的任务**。先读原任务工作的数据、训练、推理和评估代码，以它的任务定义为依据；不能只下载数据后另定一套任务并宣称完成适配。
- Capsaicin 的任务参考是 `vendor/micro_world` 中固定版本的 AMD Micro-World 2。模型及预训练参数来自 `vendor/robo_jepa` / `vendor/vjepa2`，不改用 Micro-World、Wan 或其他世界模型的生成器 checkpoint。
- 新子实验必须记录上游仓库/commit/具体文件、直接复用的部分、RoboJEPA 必需的适配，以及保留的差异及理由。优先调用上游可复用代码；重写接口时增加与上游的数值对照，不用只验证自家实现的测试替代。
- 特别核对动作单位/正负号/时间索引、视频采样、标签修正、条件输入、冻结范围、训练目标和评估协议。JEPA 与 diffusion/FM 的目标不能机械互换；需要修改时明确依据。
- 潜在空间 loss、工程 gate、短测属于前置验证，不能替代原任务的视频生成、动作可控性和视频质量评估，也不能把 latent predictor 称为已完成的交互视频生成系统。
- 优先复用权重、冻结大部分参数、独立单卡运行和共享特征缓存。已有训练预算不自动扩大，不自行增加正式 run。
- 每个子任务单独使用 W&B project；Capsaicin 为 `yibozhong657-none/robojepa-capsaicin`。只有正式训练创建 run，数据准备和验证只保留本地结果。
- 代码与小型配置/结果可提交本仓库；凭据、媒体、特征和 checkpoint 不进入 Git。上游子模块保留各自许可证。
- 集群/AP、凭据和停止规则继续遵守父目录工作区规则。状态、授权、job/PID 写入 `/home/yzhong239/RESEARCH_HISTORY.md`，不要堆入本文件。
