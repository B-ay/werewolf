# 狼人杀 Agent Team 架构

## 目标

系统以可复现的状态机驱动完整狼人杀对局。每个玩家 Agent 只能看到由引擎按身份和时点投影出的 `Observation`，所有行动都必须经过规则层校验。引擎以事件溯源形式记录事实，前端、评测器和复盘器只消费日志，不直接修改对局状态。

```text
唯一真相 GameState
       │ 私有投影（身份 / 阵营 / 时点）
       ▼
Observation ──► Role Agent ──► Intent
                                │ 白名单、目标、次数、存活校验
                                ▼
                         Rules / Reducer
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
            新 GameState                 结构化 Event Log
                                                │
                                  ┌─────────────┼─────────────┐
                                  ▼             ▼             ▼
                               观战 UI       量化评测        复盘归因
```

## 信息隔离

- 狼人只额外获得存活狼队友列表与夜间同伴意图。
- 预言家只获得自己的历史查验结果；查验结果不进入公共状态。
- 女巫只获得当夜袭击目标、药剂库存和自己的历史用药。
- 猎人只知道自身开枪权是否有效；被女巫毒杀时不可开枪。
- 普通玩家只看到公开发言、公开投票、死亡结果和已公开的身份。
- 警长是公开附加身份，不改变任何角色的私有视角；竞选报名、退水选择和选票在批次结束前只存在于裁判缓冲区。首夜待死亡名单同样不会在竞选结束前进入玩家状态、事件或进行中评测。
- HTTP API 默认返回本机裁判观战视角；玩家视角必须显式传入玩家 ID，且由引擎重新生成，不能在前端过滤全量状态。
- 非全知投影不包含角色牌顺序和随机种子；过滤后的事件流重新编号，嵌套载荷深拷贝，避免序号缺口、可预测洗牌和引用篡改三类旁路。
- 夜间人类私有行动对无权视角仍表现为“夜间进行中”；未完局的角色指标、狼人协同、查验利用率等评测只对裁判开放。

Agent 接口不接收 `GameState`，只接收与权威数据深度分离的观察快照。规则层再次验证 Agent 输出，因此即便自定义模型返回越权目标或非法动作，也只会产生修复/拒绝事件而不会污染状态。

同样的边界也约束 Agent 内部记忆：`RoleMemory` 只能消费当前玩家收到的 `Observation`，不能保存引擎真值对象的引用。角色目标、记忆快照、角色提示词和规划器中间结果属于 Agent 私有运行态，不进入公共事件日志；公开日志只包含规则允许暴露的最终行为及结果。

这里的“严格隔离”指数据面与模型上下文边界。进程内 Python Agent 是受信插件；同进程恶意代码天然可以通过反射或模块全局绕过任何对象级封装。需要运行不可信 Agent 时，应将其放入独立进程/容器，仅暴露序列化的 `Observation → Action` RPC。HTTP 服务默认仅监听回环地址，`observer` 是本机裁判能力且没有账号鉴权，不应直接发布到不可信网络。

## 回合状态机

默认流程为：初始化 → 首夜狼人协商/袭击 → 预言家查验 → 女巫行动 → **暂存首夜死亡** → 警长报名 → 候选发言 → 密封退水 → 警长投票/必要时竞选 PK → 黎明统一公布 → 警徽与猎枪处理 → 警长选择发言方向 → 白天讨论（警长最后总结归票）→ 放逐投票/必要时放逐 PK → 警徽与猎枪处理 → 胜负判定。后续夜晚不再竞选，直接在夜间技能结算后公布结果。

警长由 `GameConfig.enable_sheriff` 控制，默认开启。它是覆盖在秘密角色上的公开 badge，不是第六类 Agent：狼人、预言家、女巫、猎人、村民都能参选、持徽和传徽。候选发言以 seed 控制的随机候选为起点按座位顺时针进行；原警下玩家在竞选与 PK 两轮都有投票权，原警上玩家（包括退水者）两轮都没有投票权。二轮仍平票、无人投票、无人上警或全部退水都会使警徽永久流失。

持徽者每天以自己的座位为锚选择顺/逆时针，其他存活玩家依次发言，警长固定最后总结并提交归票目标；同一天的多轮讨论复用该顺序。放逐计票内部使用半票整数单位（普通票 `2`，警长票 `3`），避免浮点比较。首次平票者按最后一轮发言相对顺序倒序上 PK，PK 玩家禁投，其余玩家必须投有效候选；二次平票直接进入夜晚。关闭警长规则时才恢复 `vote_tie_policy`。

警长死亡时先处理警徽，再处理其猎人技能。警徽只能传给当时真正存活且不在同批死亡名单中的玩家，也可以永久撕毁；猎枪若击中新警长，会再次先处理警徽。预言家可在公开发言 metadata 中声明单/双夜 `badge_flow`，狼人也可用相同结构悍跳。该声明是可观测、可复盘的策略承诺，不是规则引擎强制执行的契约。

每次 `step()` 只推进一个可观察阶段，便于前端逐阶段演示、断点调试和人类玩家插入行动；`run()` 只是反复调用 `step()` 的便捷封装。

## 事件模型

事件至少包含：

- `seq`：游戏内严格递增序号；
- `round`、`phase`、`event_type`；
- `actor_id`、`target_id`、`message` 和经过脱敏的 `data`；
- `visibility`：`public`、具体玩家 ID 或 `observer`（只在全知持久化日志中保留）。

持久化时保留完整审计日志；向玩家或观战端分发时按视角过滤。固定随机种子、配置和 Agent 版本后，对局决策可重放。

警长相关公开事件包括 `sheriff_election_started`、`sheriff_candidacy_result`、`sheriff_speech`、`sheriff_withdrawal_result`、`sheriff_vote_started`、`sheriff_vote_result`、`sheriff_pk_started`、`sheriff_elected`、`sheriff_election_void`、`sheriff_order`、`sheriff_summary`、`badge_transfer_required`、`badge_transferred`、`badge_destroyed`、`day_pk_started` 和 `day_pk_speech`。报名、退水和投票的单人决策只写入裁判可见的 `sheriff_run_choice`、`sheriff_withdraw_choice`、`sheriff_vote_cast` 审计事件，后行动 Agent 与普通观众无法看到；批次结束后才按 `reveal_vote_choices` 向公共视角输出完整票型（含弃票）或仅汇总。

状态响应始终包含公开 `sheriff` 对象：`enabled`、`badge_status`、`holder_id`、`candidates`、`withdrawn`、`pk_candidates`、`vote_round`、`speech_order`、`day_direction`、`day_speech_order` 和 `nomination_target`；玩家投影另含 `is_sheriff`。该对象永不包含查验真值或待公布死亡。

## Agent 分层

系统不是用一个通用 Agent 在内部按 `role` 写条件分支，而是提供五个具体实现：`WerewolfAgent`、`SeerAgent`、`WitchAgent`、`HunterAgent` 和 `VillagerAgent`。`default_agent_factory` 按身份直接构造对应类型，引擎仍只依赖统一的 `decide(observation, rng)` 协议。

每个角色实例由四个相互独立的部件组成：

1. `RoleGoal`：该角色自己的目标对象，描述阵营胜利目标、能力约束和生存/信息收益偏好；每个 Agent 实例单独持有，不与同角色的其他座位共享可变对象。
2. `RoleMemory`：同一局内跨回合存在的长期记忆，只从该 Agent 收到的 `Observation` 增量更新。它可累积公开发言、票型以及该角色依法可见的私有事实，并提供显式导出/恢复能力。
3. 角色专属 prompt：定义身份、目标、允许使用的信息和输出契约。它不仅是文档元数据；接入模型时会作为 system 消息实际传给 Provider。
4. 独立 `RolePlanner`：五个角色分别实现规划逻辑，负责把目标、记忆与当前合法动作合成为 Intent。Provider 缺失、调用失败、格式错误或越权时，角色 Agent 会安全回退到自己的本地规划器。

```text
                    仅该玩家的 Observation
                              │
                  ┌───────────▼───────────┐
                  │ RoleMemory 跨回合更新 │
                  └───────────┬───────────┘
                              │
 RoleGoal ───────┐            │            ┌──── 角色专属 system prompt
                 └──────► RolePlanner ◄────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             本地确定性规划       可选 LLM Provider
                    └─────────┬─────────┘
                              ▼
                     合法性检查后的 Intent
```

五类规划器的关注点分别是：狼人协商击杀与伪装，预言家查验与公开信息时机，女巫有限药剂的救/毒收益，猎人的放逐压力与合法开枪选择，以及村民基于纯公共证据的发言和投票。不同角色的记忆字段和计划都受其观察边界约束，不能因“长期记忆”获得上一时点从未看到的信息。

五类规划器共同识别警长合法动作。预言家本地策略默认上警、不退水，公布真实首验并声明双夜警徽流；最低座次存活狼人作为默认悍跳者并在 `RoleMemory.strategy_state` 中维持一致假查验与假警徽流，其他狼人警下支持队友；女巫、猎人和村民本地基线默认不上警。外部 LLM 与人类玩家不受这些基线偏好限制，仍可自由参选。持徽者选序时优先让可疑玩家靠前、可信玩家靠后，并在最后归票当前最可疑目标。

`RoleMemory` 的默认生命周期是一局，而不是跨比赛自动共享。研究者可以显式导出并恢复记忆来开展连续学习，但必须在外层按玩家、对局、角色和 Agent 版本分区，避免把上一局身份真值错误带入下一局。导出的记忆也属于私有研究数据，不应拼入公共日志或发送给其他座位。

默认配置完全离线，使用本地规划器；在配置、随机种子和 Agent 版本固定时保持可复现。注入 LLM Provider 后，角色 system prompt 会参与真实调用，但对局能否逐动作复现还取决于 Provider 是否固定模型、采样参数并提供确定性响应。无论使用哪种模式，规则引擎都会对最终 Intent 做相同校验。

这种划分允许分别替换某一个角色的目标、记忆、提示词、规划器或模型，在完全相同的规则、观察边界和随机种子下进行消融实验与同台竞技。

## DeepSeek Provider 与密钥边界

服务端启动时调用 `build_deepseek_provider_from_env()` 读取 `DEEPSEEK_API_KEY`；`DEEPSEEK_MODEL` 为可选项，默认使用当前的 `deepseek-flash`，也可以显式选择 `deepseek-v4-pro`。Provider 使用 DeepSeek 当前官方 OpenAI 兼容基础地址 [`https://api.deepseek.com`](https://api-docs.deepseek.com/guides/codex)，具体请求发送到 `https://api.deepseek.com/chat/completions`。环境变量在进程启动时读取，因此新增或修改配置后需要重启服务。

```text
服务器环境变量 DEEPSEEK_API_KEY
              │（仅启动时读取）
              ▼
       DeepSeekProvider ───────────────► https://api.deepseek.com
              ▲                                  │
              │ 角色 prompt + 当前 Observation   │ 模型结果 / 脱敏错误
              │                                  ▼
          五类 Role Agent ──合法性校验──► Intent
              │
              └── 请求失败或结果非法 ──► 对应角色的本地 RolePlanner

浏览器 ◄── 仅 configured 布尔值与模型名 ── ArenaManager
```

浏览器只收到 Provider 是否配置以及非敏感的模型名称。密钥不会进入前端资源、HTTP 响应、`GameState`、`Observation`、事件日志、评测结果或回放；网络异常和服务端错误也只以脱敏消息暴露。角色 Agent 获得的是共享 Provider 引用，而不是密钥字符串，并继续只向模型提供该角色依法可见的观察、长期记忆摘要和角色专属 system prompt。

当 Provider 已配置时，创建对局 UI 会自动启用并默认选择 DeepSeek；未配置时该选项禁用。DeepSeek 策略下每个 AI 决策都是一次可计费模型调用，自动播放或纯 AI 对局会快速累积调用次数。调用失败、超时、响应无法解析或动作不合法时，失败只影响当前决策，Agent 随即使用自己的本地规划器生成合法动作；这项可用性回退不代表远端调用免费。本地 `heuristic` 策略完全不调用 DeepSeek。

Windows PowerShell 的最小启动方式如下；示例故意只使用占位符，真实密钥不应写入仓库或任何配置文件：

```powershell
$env:DEEPSEEK_API_KEY="<YOUR_DEEPSEEK_API_KEY>"
$env:DEEPSEEK_MODEL="deepseek-flash"  # 可选，也可设为 deepseek-v4-pro
python run.py
```

## 测评与复盘

评测器输出 `1.1` schema 的三层结果：

- 结果层：阵营胜负、角色存活、局长、淘汰顺序。
- 过程层：放逐票命中率、分别计算的狼刀/放逐/警长票协同、竞选结果、警长领导力、1.5 票决定性、归票响应率、警徽传承、警徽流执行、查验利用率、能力收益、言票一致性和疑似私有信息泄露等。
- 归因层：串联“竞选—当选/流局—发言顺序—归票—传徽/撕徽”，并结合死亡、关键技能、票型翻转和胜负边界生成可回溯到事件序号的中文结论。

新指标暂不进入原有贡献综合分或 Elo，保证与旧实验纵向可比。评测器对缺少 `sheriff` 字段和新事件的旧回放返回空样本/`None`，而不是把缺失误判为零表现。

榜单对 Agent 版本维护场次、胜率、分角色胜率与 Elo。Elo 更新以一局中的阵营对抗为单位，避免简单胜率被角色分配偏差主导。

## 扩展点

- 新角色：实现能力阶段、观察投影与合法动作校验，并注册到角色表。
- 新 Agent：实现统一决策协议，不需要修改引擎。
- 新模型：实现模型适配器，记录 `provider/model/prompt_version` 作为 Agent 版本。
- 新规则：通过 `GameConfig` 配置人数、角色表、平票与警长规则；复杂变体可新增状态阶段。
- 批量竞技：按种子和座位轮换生成赛程，日志交由同一个评测器汇总。
