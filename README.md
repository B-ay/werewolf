# NOCTURNE · 狼人杀多智能体评测平台

基于 Python 标准库实现的多智能体对战与评测系统，提供规则状态机、按玩家视角生成的 Observation、模型服务接入与失败回退、HTTP API、观战界面和批量竞技。支持纯本地启发式运行，也可接入 DeepSeek。

本实现提供进阶方向 **2：测评 + 复盘**，并加入方向 **3：基于验证的策略自进化**。

## 工程亮点

| 能力 | 实现与边界 |
| --- | --- |
| 规则与决策解耦 | 规则引擎控制阶段、合法动作与胜负；五类角色 Agent 通过统一协议决策 |
| 信息隔离 | 按玩家投影视角、深拷贝嵌套数据、过滤私有事件并重新编号；覆盖相关回归测试 |
| 模型调用容错 | Provider 封装超时、有限重试与错误脱敏；异常或非法输出回退到本地规划器 |
| 服务与回放 | HTTP API、后台自动运行、对局锁、增量事件查询及完局 JSON 保存 |
| 批量评测 | 多随机种子、座位轮换、单局失败隔离、分角色统计与版本 Elo |
| 策略自进化 | 完局反馈、有界候选、独立种子成对验证、通过后继承；不修改模型权重 |
| 可复现基线 | 固定配置、种子及本地规划器时可复现；外部模型的确定性需单独保证 |

这是用于单机研究与演示的 Agent 应用及评测工具。当前未提供高并发压测、分布式调度、推理引擎优化或线上可用性数据。

## 导航

- [快速启动](#一分钟启动)
- [模型接入](#启用-deepseek)
- [HTTP API](#http-api)
- [批量评测](#批量竞技与榜单)
- [跨局自进化](#跨局自进化)
- [架构与信任边界](docs/ARCHITECTURE.md)
- [验证记录](#验证记录)

## 验证记录

2026-10-14，在 Windows / Python 3.14.4 环境运行以下命令，现有 **72 项测试全部通过**，覆盖规则引擎、角色 Agent、观察隔离、Provider、HTTP API、警长流程、评测与批量竞技。

```sh
python -m unittest discover -s tests -v
```

上述历史验证的 Provider 测试使用模拟请求，当时未调用真实付费模型，也未进行浏览器视觉验收或并发性能测试。

2026-10-15，模块重构与策略自进化接入后，**120 项 Python 测试全部通过**；另完成 9 项前端同步回归与 4 组桌面/移动端真实后端烟测，覆盖纯 AI、人机混战、经验开关、复盘与榜单。两轮离线实验共运行 6 局训练对局，验证 10 个候选、240 对新旧策略对照局；候选均未达到晋级门槛，因此保留原策略。未调用真实付费模型，未进行并发性能测试，测试通过不等同于生产部署验证。

## 一分钟启动

只需要 Python 3.10+，运行时无第三方依赖：

```bash
python run.py
```

浏览器打开 <http://127.0.0.1:8000>。在创建面板中选择人数、随机种子和可选的人类座位，即可逐步执行或自动播放。

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

## 启用 DeepSeek

DeepSeek 密钥只由服务端从环境变量读取，创建对局时不需要、也不允许在浏览器中填写密钥。在 Windows PowerShell 中使用占位符完成配置，并从**同一个终端**启动或重启服务：

```powershell
$env:DEEPSEEK_API_KEY="<YOUR_DEEPSEEK_API_KEY>"
$env:DEEPSEEK_MODEL="deepseek-flash"  # 可选；不设置时也是该默认值
python run.py
```

如需使用能力更强、成本也可能更高的模型，可将 `DEEPSEEK_MODEL` 改为 `deepseek-v4-pro`。修改环境变量后必须重启已经运行的狼人杀服务，因为 Provider 只在服务进程启动时读取配置。DeepSeek 当前官方 OpenAI 兼容基础地址是 [`https://api.deepseek.com`](https://api-docs.deepseek.com/guides/codex)，本项目实际调用其 Chat Completions 地址 `https://api.deepseek.com/chat/completions`。

配置成功后，创建对局页面会自动启用并默认选择 `DeepSeek · <模型名>`；未配置时该选项保持禁用，并继续使用本地启发式 Agent。DeepSeek 策略下，每一次 AI 决策都会调用模型并按服务商的 token 规则计费，一局纯 AI 对战会产生多次调用；不希望产生费用时请选择 `Heuristic · 本地启发式`。单次模型请求失败、超时、输出格式错误或给出非法动作时，仅该次决策回退到对应角色的本地规划器，对局仍可继续。

`DEEPSEEK_API_KEY` 始终保留在服务端进程中，只作为访问 DeepSeek 官方接口的授权信息使用。服务端不会把它返回给浏览器，也不会写入游戏状态、结构化事件日志、评测结果或回放文件；错误信息同样经过脱敏。不要把密钥放进 `GameConfig`、URL、前端代码或对局发言。

## 已实现能力

- 完整角色：狼人、预言家、女巫、猎人、村民。
- 完整流程：狼人协商、预言家查验、女巫解药/毒药、首日警长竞选、警长安排发言与归票、警徽传承、猎人开枪、白天发言、放逐 PK 和胜负判定。
- 警长规则：警徽独立于角色，任意角色均可参选；支持密封报名/退水/两轮竞选票、逆序 PK、警长 1.5 票、顺逆时针发言、最后归票、死亡传徽或撕徽，以及可观测但不强制执行的单双夜警徽流。
- 五套角色 Agent：`WerewolfAgent`、`SeerAgent`、`WitchAgent`、`HunterAgent`、`VillagerAgent` 都有独立的角色目标对象、跨回合长期记忆、角色专属提示词和规划器；仍统一遵守 `Agent.decide(observation, rng)` 协议。
- 双模式决策：默认使用无需网络和密钥的本地确定性规划器；也可向任一角色 Agent 注入 LLM Provider，模型异常或返回非法动作时自动回退到该角色的本地规划器。
- 强投影隔离：Agent 永远拿不到引擎真值对象；角色、私有事件、技能资源与查验结果按视角投影，嵌套载荷深拷贝，规则层再次校验全部 Intent。
- 防侧信道日志：玩家视角过滤私有事件后重新编号，避免从日志序号缺口推断隐藏行动。
- 防过程侧信道：玩家视角隐藏洗牌种子、夜间私有等待态与进行中的全知评测；具体玩家视角不能用 `omniscient=true` 绕过投影。
- 人机混战：任意一个或多个座位可设为 `human`，引擎在该玩家行动时暂停并返回严格限定的合法动作集合。
- 可复现：配置、随机种子和 Agent 版本固定时，状态与事件流确定性复现。
- 全程可观测：事件包含轮次、阶段、行为者、目标、数据和可见范围；完局后自动保存到 `data/game-*.json`。
- 实时观战：座位环、角色/阵营/存活状态、回合阶段、发言与技能时间线、控制台、视角切换和响应式布局。
- 测评复盘：`1.1` schema 区分狼刀票、警长票和放逐票，并量化竞选、领导力、1.5 票决定性、归票响应、警徽传承、警徽流执行、言票一致性、信息泄露、查验利用和狼人协同。
- 批量竞技：多 seed、多局、座位轮换、失败隔离、分角色统计和模型/版本 Elo 榜单。

## 核心用法

### 跨局自进化

创建对局时勾选“启用跨局自进化”（API：`enable_evolution: true`）。默认关闭，
关闭时继续使用原有固定策略，不读写跨局经验。开启后按以下闭环执行：

1. **复盘反馈**：仅在完局后，利用裁判真值统计各 AI 角色的胜负、误投、误毒/误枪与言票不一致；人类动作不计入 AI 错误。将结果转为角色通用中文经验，不复制旧局身份、座位或原话。
2. **策略更新**：按失败反馈生成有界候选，调整女巫出毒阈值、猎人开枪阈值、预言家查验偏好、狼人信息角色威胁权重和公共嫌疑权重。候选不会立即替换正式版本。
3. **改进验证**：每个候选至少使用 24 个独立验证 seed，分别运行旧版与候选版，固定规则、座位分配和其余角色版本。要求胜率差至少 +5 个百分点、角色贡献分下降不超过 0.02、无信息泄露/引擎动作修复、模型降级率不增加，并通过精确符号检验（同轮多候选校正）。不足或未达标保持旧版。
4. **经验继承与再对局**：仅晋级版本落盘；下一局为每个座位创建全新记忆，注入该角色的策略参数和白名单经验。进行中的对局保持创建时的版本快照。模型与规则板型分别建库，本地验证结果不能冒充 DeepSeek 改进。

这里进化的是**策略参数与提示词中的通用经验**，不是微调 DeepSeek 权重，也不让模型改写源代码。
候选由确定性规则提出，复盘属于规则归因，不是自动发现任意新算法；验证通过表示本次实验有提升证据，不保证此后每局获胜。

本地模式完局后自动在后台验证；DeepSeek 模式只自动记录反馈，不额外启动付费批量对局。
可通过 [`/api/evolution`](http://127.0.0.1:8000/api/evolution) 查看后台状态、正式版本、候选参数、反馈快照和验证结果。
重启后会继续使用 `data/evolution/strategies.json` 中已晋级的版本，该目录已忽略 Git 跟踪，文件不包含密钥。
文件使用线程锁、跨进程文件锁和原子替换，锁冲突会明确报错，不会覆盖另一个进程的版本。

独立运行两轮“对局—反馈—优化—验证—再对局”：

```bash
python -m backend.evaluation.evolution --seeds 5,42,31415 --rounds 2 --validation-games 24 --output data/evolution/report.json
```

训练种子与验证种子严格分开，验证结果不会回灌为训练反馈；报告保留候选版本、对手版本、成对结果、胜率差与统计检验结果。
可以用 `--store <其他路径>` 隔离实验库，避免实验影响实时对局。`--validation-games` 小于 24 只用于调试，永远不会晋级。

DeepSeek 批量实验必须显式确认费用（会产生大量模型调用，需先配置服务端环境变量）：

```bash
python -m backend.evaluation.evolution --deepseek --allow-paid-validation --seeds 5,42,31415 --rounds 1 --validation-games 24 --output data/evolution/deepseek-report.json
```

### 引擎接口

```python
from backend.game.models import GameConfig
from backend.game.engine import WerewolfGame

game = WerewolfGame(GameConfig(player_count=8, seed=42))
final_state = game.run()

print(final_state["winner"])
print(game.event_log(omniscient=True))
```

`GameConfig.enable_sheriff` 默认是 `True`。如需复现实装警长前的传统流程，可显式传入 `enable_sheriff=False`；此时白天平票继续遵循 `vote_tie_policy`。警长规则开启时，首次放逐平票固定进入 PK，二次平票无人出局。

注入自定义 Agent：

```python
game = WerewolfGame(
    GameConfig(seed=42),
    agent_factory=lambda player_id, role: MyAgent(player_id, role),
)
```

Agent 只能看到：

```python
class MyAgent:
    def decide(self, observation, rng):
        # observation.self_state / players / events / legal_actions / private_facts
        # 返回 Action 或同结构字典
        ...
```

五类角色通过 `BaseAgent` 共用很小的 `LLMProvider.complete(messages, temperature=...)` 协议接入任意模型服务。模型输出越权、格式错误或异常时会降级为合法的启发式动作，保证对局不被单个 Agent 中断。

## 五套独立角色 Agent

`create_agent(player_id, role)` 不再为所有身份创建同一个、再按角色分支的通用实现，而是直接实例化对应的具体角色类：

| 角色实现 | 独立目标 | 记忆与规划重点 |
|---|---|---|
| `WerewolfAgent` | 让狼人阵营达到胜利条件，同时维持自身与狼队的伪装 | 只记录观察中可见的狼队友、夜间协商、公开身份主张和票型；最低座次存活狼人默认悍跳并维持一致的假查验/假警徽流 |
| `SeerAgent` | 通过查验定位狼人，并在信息收益与暴露风险间取舍 | 默认上警且不退水，公开真实首验并规划双夜警徽流；死亡时根据已完成查验决定传徽或撕徽 |
| `WitchAgent` | 在有限药剂约束下提高好人阵营存活与击杀收益 | 记录自己可见的夜袭目标、药剂库存和用药历史；独立规划救、毒或保留药剂 |
| `HunterAgent` | 为好人阵营建立放逐压力，并在合法触发时选择开枪目标 | 根据公开发言、票型和死亡信息更新嫌疑；独立规划发言、投票与开枪/不开枪 |
| `VillagerAgent` | 仅凭公共证据识别并放逐狼人 | 长期累积公开发言、身份主张、警务状态和票型；规划质询、表态与投票 |

每个实例都拥有自己的 `RoleGoal`、`RoleMemory` 和角色规划器，彼此不共享可变状态。`RoleMemory.strategy_state` 只保存该角色从合法视角得出的持续策略状态，例如预言家的警徽流计划或悍跳狼的一致假查验；不会引入其他角色私有事实。这里的“长期记忆”默认指**同一局内跨回合持续存在**：每次决策只从该玩家收到的 `Observation` 更新记忆，不读取 `GameState` 或其他玩家内存。记忆支持显式 `export()` / `restore()`，用于研究者自行实现跨局实验；跨局恢复时应按玩家、对局、角色和 Agent 版本分区，系统不会默认把上一局知识带入下一局。

五类 Agent 均带角色专属 system prompt。未注入 Provider 时，提示词不会触发外部调用，决策由本地规划器完成；注入 Provider 后，该角色提示词会实际作为 system 消息发送，模型输出仍须通过合法动作校验。固定配置、随机种子和本地规划器可确定性复现；使用外部模型时，还需要 Provider 本身采用固定模型、固定参数并提供确定性保证，才能声称相同复现性。

局内 Agent 的公开发言、警长竞选、PK 发言和归票统一使用简体中文，本地基线与
DeepSeek 提示词采用相同语言要求。模型返回纯英文正文时，会使用该角色已经生成的
中文本地基线；玩家名称、ID 和 JSON 字段/枚举值保留原格式。狼队夜票与技能反馈
文案也使用中文，原有可见范围保持不变。已保存的历史对局和人类输入保留原文。

角色目标、记忆快照、system prompt 和内部规划过程不会写入公共事件日志。日志只记录规则允许公开的发言、动作结果与状态变化；私有行动仍由原有可见范围和观察投影控制。

## 人类玩家接管

```python
game = WerewolfGame(GameConfig(seed=7, human_player="p3"))
state = game.run()                 # 在 p3 需要行动时暂停
pending = state["pending_action"] # 只包含该视角可用的合法动作
legal = pending["legal_actions"][0]
action = {
    "actor_id": "p3",
    "action_type": legal["action_type"],
}
if legal["target_ids"]:
    action["target_id"] = legal["target_ids"][0]
if legal["content_required"]:
    action["content"] = "我会基于公开发言和票型继续判断。"

game.submit_action(action)
game.run()
```

也可使用网页创建面板选择人类座位；公开观战视角与玩家私有视角由后端分别生成。

> 安全边界：默认服务只监听 `127.0.0.1`，定位为单机研究/裁判控制台。内置 `observer` 是有意保留的全知裁判能力，HTTP 层不含多人账号鉴权；不要把端口直接暴露到不可信网络。进程内自定义 Python Agent 也属于受信插件（它理论上可用反射读取进程内存）。接入不可信 Agent 时，应把它放到独立进程或容器，只经序列化 `Observation → Action` 通道通信。

## HTTP API

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/health` | 服务状态 |
| `POST` | `/api/games` | 创建对局，正文为 `GameConfig` |
| `GET` | `/api/games/{id}` | 获取状态，可传 `viewer`、`omniscient` |
| `POST` | `/api/games/{id}/step` | 推进一步 |
| `POST` | `/api/games/{id}/run` | 后台自动播放 |
| `POST` | `/api/games/{id}/stop` | 暂停自动播放 |
| `POST` | `/api/games/{id}/human-action` | 提交当前人类动作 |
| `GET` | `/api/games/{id}/events?after=N` | 增量事件流 |
| `GET` | `/api/games/{id}/evaluation` | 过程评测与复盘；进行中玩家视角返回脱敏占位 |
| `GET` | `/api/replays` | 对局列表 |
| `GET` | `/api/leaderboard` | 当前服务实例榜单 |

网页默认使用 HTTP 状态同步，`/api/health` 的 `transport` 字段声明
`http_polling` 和默认 2200 毫秒的同步间隔。状态接口会在同一把锁内返回状态与
可见事件，前端每轮只请求一次快照；请求完成后才安排下一轮，终局后停止轮询。
各区域仅在其展示数据变化时更新，座位、日志和发言条目按稳定 ID 复用。只有读者
原本在列表底部时才跟随新条目，阅读历史时保留位置；切换视角会清除上一视角内容。

浏览器回归测试覆盖重复快照、节点复用、滚动锚点、人类发言草稿和视角隔离。
在已安装 Playwright 且可使用 Microsoft Edge 的 Node 环境中执行：

```bash
node tests/frontend_sync.spec.cjs
node tests/frontend_backend.spec.cjs
```

第二项测试启动随机端口的独立后端，在桌面和移动布局下完成纯 AI／人机混战，
检查警长、玩家视角、真人操作与完局复盘；回放只写临时目录，不调用真实模型。
其他 Chromium 通道可通过 `PLAYWRIGHT_CHANNEL` 指定，后端测试可用 `PYTHON`
指定 Python 可执行文件。

## 批量竞技与榜单

```bash
python -m backend.evaluation.tournament \
  --seeds 1,2,5-8 \
  --games-per-seed 2 \
  --player-count 8 \
  --competitor baseline:heuristic:v1 \
  --competitor candidate:my-model:v2 \
  --output tournament.json
```

Python API：

```python
from backend.evaluation.tournament import run_tournament

report = run_tournament(
    seeds=[1, 2, 3],
    games_per_seed=2,
    game_config={"player_count": 8},
    rotate_seats=True,
)
print(report["leaderboard"])
```

报告包含 `games`、`evaluations`、`leaderboard`、`summaries` 和 `failures`。Elo 以阵营间成对结果做零和更新，并保留角色维度胜率，降低身份分配造成的偏差。

评测只服务本项目：单局入口为 `evaluate_game(events, final_players, winner)`，
也可直接传入保存的回放对象；榜单入口为 `EloLeaderboard` 和
`build_leaderboard()`。兼容本项目无警长字段的历史回放，不再猜测外部日志字段、
事件名称或游戏引擎接口。单局 Schema 1.1、竞技报告 Schema 1.0 和现有展示字段
保持不变。

自定义参赛 Agent 统一使用 `factory(player_id, role)`，每次返回独立的
`decide(observation, rng)` 实例；不再接受零参数／单参数工厂或 Agent 原型对象。
测试注入的游戏工厂必须接受 `config, agent_factory=...`，返回支持
`run(max_steps=...)`、`result()`、`event_log(omniscient=True)` 的对象。

## 项目结构

```text
backend/
  agents/
    base.py       BaseAgent：视角/记忆/上下文/模型输出校验与降级
    memory.py     独立角色目标、长期记忆、容量裁剪与恢复
    planner.py    公开证据评分、目标选择及发言/投票/警长规划
    factory.py    唯一 create_agent 入口、角色注册与 Provider 注入
    werewolf.py   狼人 Agent 与狼人专属 Planner
    seer.py       预言家 Agent 与查验/警徽流 Planner
    witch.py      女巫 Agent 与药剂 Planner
    villager.py   村民 Agent 与公开推理 Planner
    hunter.py     猎人 Agent 与开枪 Planner
  llm/
    base.py       通用 LLMProvider 协议
    provider.py   DeepSeek 请求、重试、脱敏与环境配置
  game/
    models.py     角色、阶段、动作、观察、配置等领域契约
    engine.py     信息隔离、警长规则及完整对局状态机
  logger/
    events.py     事件写入、可见性过滤和安全序号投影
    storage.py    回放 JSON 原子落盘
  evaluation/
    common.py     本项目日志契约、一次性事件索引与共享统计
    metrics.py    信息隔离、言票一致性、协同；竞选/领导力/警徽流分步评测
    replay.py     转折点与中文复盘归因
    evaluator.py  Schema 1.1 单局评测入口
    leaderboard.py Elo 和角色维度排行榜
    tournament.py 批量竞技与 CLI
  server.py       HTTP/静态服务，保留在 backend 根目录
frontend/
  index.html      观战控制台结构
  styles.css      响应式视觉系统
  app.js          实时交互与状态渲染
tests/             引擎、评测、API、竞技测试
docs/ARCHITECTURE.md
```

更详细的信任边界、状态机、事件模型和扩展方式见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

五类角色均继承 `backend.agents.base.BaseAgent`，对外统一调用
`decide(observation, rng)`。基类内部负责 `update_state()`、上下文构造、
本地规划、模型输出解析与校验、降级和最终动作记录；每次决策只摄取一次记忆、
记录一次最终动作。公共策略算法放在 `RolePlanner`，各角色继续拥有独立目标、
记忆、专属提示词和规划器。

旧兼容入口、`RoleAgent` 别名和通用旧 Agent 已移除，请使用明确的职责模块：

```python
from functools import partial
from backend.agents.factory import create_agent
from backend.llm.provider import DeepSeekProvider

provider = DeepSeekProvider.from_env()
game = WerewolfGame(
    GameConfig(seed=42),
    agent_factory=partial(create_agent, provider=provider),
)
```

未注入 Provider 时使用本地规划；共享 Provider 不共享任何座位的记忆。
批量竞技入口为 `python -m backend.evaluation.tournament`。
已有 HTTP 接口、回放格式和记忆快照保持兼容。

## 研究扩展建议

1. 用真实 LLM Provider 替换部分座位，记录 `provider/model/prompt_version` 后做同 seed、轮换座位的对照实验。
2. 将评测器的转折点作为训练样本，形成“对局 → 归因 → 策略/提示词优化 → 再对局”循环。
3. 为发言增加结构化 belief/claim 图，进一步量化欺骗、一致性、信息增益和阵营协同。
4. 新角色只需增加观察投影、合法动作、阶段处理与注册项；无需修改现有 Agent 协议。
