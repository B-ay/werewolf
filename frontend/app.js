(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const query = new URLSearchParams(window.location.search);
  const API_BASE = (query.get("api") || (window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "")).replace(/\/$/, "");
  const HISTORY_KEY = "nocturne.match-history.v1";
  const MAX_EVENTS = 500;

  const ROLE_META = {
    werewolf: { name: "狼人", icon: "♞", team: "wolf" },
    wolf: { name: "狼人", icon: "♞", team: "wolf" },
    seer: { name: "预言家", icon: "◈", team: "good" },
    witch: { name: "女巫", icon: "⚗", team: "good" },
    hunter: { name: "猎人", icon: "⌖", team: "good" },
    villager: { name: "平民", icon: "♙", team: "good" },
    village: { name: "好人", icon: "♙", team: "good" },
    unknown: { name: "身份未知", icon: "?", team: "unknown" },
  };

  const PHASE_META = {
    setup: { name: "初始化", title: "角色正在入场", subtitle: "规则引擎正在完成身份分配与私有观察投影。", icon: "✦", track: 0 },
    night: { name: "夜间", title: "夜幕降临", subtitle: "特殊角色依次行动，私有信息仅向对应 Agent 开放。", icon: "◑", track: 1 },
    night_wolf: { name: "狼人行动", title: "狼群正在密谋", subtitle: "狼人 Agent 在隔离频道内协商今夜的袭击目标。", icon: "♞", track: 1 },
    wolf_action: { name: "狼人行动", title: "狼群正在密谋", subtitle: "狼人 Agent 在隔离频道内协商今夜的袭击目标。", icon: "♞", track: 1 },
    night_seer: { name: "预言家查验", title: "预言家正在查验", subtitle: "查验结果只进入预言家的私有观察，不写入公共状态。", icon: "◈", track: 1 },
    seer_action: { name: "预言家查验", title: "预言家正在查验", subtitle: "查验结果只进入预言家的私有观察，不写入公共状态。", icon: "◈", track: 1 },
    night_witch: { name: "女巫行动", title: "女巫正在抉择", subtitle: "解药与毒药均受库存及单夜行动规则约束。", icon: "⚗", track: 1 },
    night_witch_save: { name: "女巫解药", title: "女巫正在抉择", subtitle: "是否使用解药，将直接改变黎明时的死亡结算。", icon: "⚗", track: 1 },
    night_witch_poison: { name: "女巫毒药", title: "女巫正在抉择", subtitle: "毒药目标由规则层再次校验，避免越权行动。", icon: "⚗", track: 1 },
    sheriff_run: { name: "上警报名", title: "警长竞选开始", subtitle: "所有玩家同时决定是否上警，选择将在收齐后统一公开。", icon: "♛", track: 2 },
    sheriff_speech: { name: "警上发言", title: "候选人正在竞选", subtitle: "警上玩家依序发言，陈述身份、查验与警徽流安排。", icon: "♛", track: 2 },
    sheriff_withdraw: { name: "退水阶段", title: "候选人确认退水", subtitle: "退水后不再拥有本轮被投票权与投票权。", icon: "↘", track: 2 },
    sheriff_vote: { name: "警长投票", title: "警下玩家正在投票", subtitle: "只有原警下玩家拥有投票权，警徽将授予最高票候选人。", icon: "♛", track: 2 },
    sheriff_pk_speech: { name: "警长竞选 PK", title: "平票候选人再次发言", subtitle: "平票玩家按首次发言顺序倒序进行 PK 发言。", icon: "⇄", track: 2 },
    sheriff_pk_vote: { name: "警长 PK 投票", title: "警徽归属再次表决", subtitle: "若本轮再次平票，本局警徽将永久流失。", icon: "♛", track: 2 },
    night_resolve: { name: "昨夜结算", title: "天光即将揭晓", subtitle: "夜间结果已结算，正等待竞选完成后统一公布。", icon: "☀", track: 2 },
    dawn_reveal: { name: "公布结果", title: "天光揭晓结果", subtitle: "昨夜的死亡与公开身份现在向所有玩家公布。", icon: "☀", track: 3 },
    dawn: { name: "公布结果", title: "天光揭晓结果", subtitle: "昨夜的死亡与公开身份将按规则向所有玩家公布。", icon: "☀", track: 3 },
    sheriff_badge: { name: "处理警徽", title: "警长正在处理警徽", subtitle: "死亡警长可将警徽移交给存活玩家，或撕毁警徽。", icon: "♛", track: 3 },
    sheriff_order: { name: "决定发言顺序", title: "警长正在安排发言", subtitle: "警长以自身座位为锚选择顺时针或逆时针发言。", icon: "↻", track: 4 },
    day: { name: "白天", title: "白日议事", subtitle: "存活玩家基于公开信息更新判断。", icon: "☀", track: 4 },
    day_discussion: { name: "白天发言", title: "白日议事", subtitle: "玩家依警长指定顺序发言，警长最后总结并归票。", icon: "“", track: 4 },
    discussion: { name: "白天发言", title: "白日议事", subtitle: "每位 Agent 正在陈述推理、回应质疑并建立公开承诺。", icon: "“", track: 4 },
    day_vote: { name: "放逐投票", title: "放逐投票进行中", subtitle: "警长拥有 1.5 票，其余存活玩家各拥有 1 票。", icon: "⌁", track: 5 },
    vote: { name: "放逐投票", title: "放逐投票进行中", subtitle: "公开发言正在转化为票型，规则层将处理平票与无效票。", icon: "⌁", track: 5 },
    day_pk_speech: { name: "放逐 PK 发言", title: "平票玩家正在 PK", subtitle: "平票玩家按最后发言顺序倒序陈述，其余玩家随后重新投票。", icon: "⇄", track: 5 },
    day_pk_vote: { name: "放逐 PK 投票", title: "放逐票型再次表决", subtitle: "PK 玩家不得投票；若再次平票，本日无人出局。", icon: "⌁", track: 5 },
    hunter_shot: { name: "猎人开枪", title: "猎人的最后抉择", subtitle: "开枪权由死亡原因与规则配置共同决定。", icon: "⌖", track: 5 },
    finished: { name: "对局结束", title: "终局已裁定", subtitle: "完整事件链已冻结，可进入复盘中心查看归因。", icon: "♜", track: 6 },
  };

  const EVENT_META = {
    speech: { icon: "“", label: "公开发言", category: "speech" },
    speak: { icon: "“", label: "公开发言", category: "speech" },
    statement: { icon: "“", label: "公开发言", category: "speech" },
    vote: { icon: "⌁", label: "投票", category: "vote" },
    wolf_kill: { icon: "♞", label: "狼人袭击", category: "danger" },
    kill: { icon: "♞", label: "袭击", category: "danger" },
    death: { icon: "†", label: "玩家死亡", category: "danger" },
    eliminated: { icon: "†", label: "玩家出局", category: "danger" },
    seer_check: { icon: "◈", label: "预言家查验", category: "action" },
    check: { icon: "◈", label: "身份查验", category: "action" },
    witch_save: { icon: "+", label: "女巫使用解药", category: "action" },
    save: { icon: "+", label: "救治", category: "action" },
    witch_poison: { icon: "⚗", label: "女巫使用毒药", category: "danger" },
    poison: { icon: "⚗", label: "毒杀", category: "danger" },
    hunter_shoot: { icon: "⌖", label: "猎人开枪", category: "danger" },
    hunter_shot: { icon: "⌖", label: "猎人开枪", category: "danger" },
    game_started: { icon: "✦", label: "对局开始", category: "system" },
    game_start: { icon: "✦", label: "对局开始", category: "system" },
    role_assigned: { icon: "◆", label: "身份分配", category: "system" },
    wolf_team: { icon: "♞", label: "狼队信息", category: "action" },
    night_started: { icon: "◑", label: "夜幕降临", category: "system" },
    day_started: { icon: "☀", label: "白日开始", category: "system" },
    phase_started: { icon: "→", label: "阶段流转", category: "system" },
    phase_changed: { icon: "→", label: "阶段流转", category: "system" },
    phase_change: { icon: "→", label: "阶段流转", category: "system" },
    night_result: { icon: "☀", label: "昨夜结果", category: "system" },
    vote_started: { icon: "⌁", label: "投票开始", category: "vote" },
    game_over: { icon: "♜", label: "胜负裁定", category: "system" },
    game_end: { icon: "♜", label: "胜负裁定", category: "system" },
    game_finished: { icon: "♜", label: "胜负裁定", category: "system" },
    invalid_action: { icon: "!", label: "无效行动", category: "danger" },
    sheriff_election_started: { icon: "♛", label: "警长竞选开始", category: "sheriff" },
    sheriff_run_choice: { icon: "✋", label: "封存上警选择", category: "sheriff" },
    sheriff_candidacy_result: { icon: "♛", label: "上警结果", category: "sheriff" },
    sheriff_speech: { icon: "“", label: "警上发言", category: "sheriff" },
    sheriff_pk_speech: { icon: "“", label: "警长竞选 PK 发言", category: "sheriff" },
    sheriff_withdrawal_result: { icon: "↘", label: "退水结果", category: "sheriff" },
    sheriff_withdraw_choice: { icon: "↘", label: "封存退水选择", category: "sheriff" },
    sheriff_vote_started: { icon: "♛", label: "警长投票开始", category: "sheriff" },
    sheriff_vote_cast: { icon: "♛", label: "封存警长票", category: "sheriff" },
    sheriff_vote_result: { icon: "♛", label: "警长票型", category: "sheriff" },
    sheriff_pk_started: { icon: "⇄", label: "警长竞选 PK", category: "sheriff" },
    sheriff_elected: { icon: "♛", label: "警长当选", category: "sheriff" },
    sheriff_election_void: { icon: "♛", label: "警徽流失", category: "sheriff" },
    dawn_reveal_started: { icon: "☀", label: "公布夜间结果", category: "system" },
    sheriff_order: { icon: "↻", label: "发言顺序", category: "sheriff" },
    sheriff_summary: { icon: "♛", label: "警长归票", category: "sheriff" },
    badge_transfer_required: { icon: "♛", label: "等待移交警徽", category: "sheriff" },
    badge_transferred: { icon: "♛", label: "警徽移交", category: "sheriff" },
    badge_destroyed: { icon: "×", label: "警徽撕毁", category: "sheriff" },
    day_vote_result: { icon: "⌁", label: "放逐票型", category: "vote" },
    day_pk_started: { icon: "⇄", label: "放逐 PK", category: "vote" },
    day_pk_speech: { icon: "“", label: "放逐 PK 发言", category: "speech" },
  };

  const store = {
    game: null,
    events: [],
    eventKeys: new Set(),
    eventFilter: "all",
    overlay: "identity",
    perspective: "observer",
    socket: null,
    reconnectTimer: null,
    pollTimer: null,
    auto: false,
    runController: null,
    sound: false,
    effects: true,
    resultShownFor: null,
    connection: "connecting",
    viewRevision: 0,
    humanDraft: null,
    history: readHistory(),
    leaderboard: null,
    providers: {},
  };

  function readHistory() {
    try {
      const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      // Older builds could place an in-progress observer snapshot in history.
      // Only completed games are eligible for durable replay storage.
      if (!Array.isArray(value)) return [];
      const completed = value.filter((item) => item?.completed === true || Boolean(item?.winner));
      if (completed.length !== value.length) localStorage.setItem(HISTORY_KEY, JSON.stringify(completed.slice(0, 20)));
      return completed;
    } catch {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(store.history.slice(0, 20)));
    } catch {
      // The application remains usable when local storage is disabled.
    }
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function firstDefined(...values) {
    return values.find((value) => value !== undefined && value !== null);
  }

  function clamp(value, minimum = 0, maximum = 100) {
    return Math.min(maximum, Math.max(minimum, Number(value) || 0));
  }

  function percent(value) {
    if (value === null || value === undefined || value === "") return null;
    if (typeof value === "object") {
      value = firstDefined(value.score, value.rate, value.ratio, value.value, value.accuracy, value.consistency_rate, value.utilization_rate, value.coordination_score);
    }
    const number = Number(value);
    if (!Number.isFinite(number)) return null;
    return clamp(number <= 1 ? number * 100 : number);
  }

  function booleanValue(value, fallback = false) {
    if (value === undefined || value === null) return fallback;
    if (typeof value === "string") return !["false", "0", "no", "off", ""].includes(value.toLowerCase());
    return Boolean(value);
  }

  function normalizeTeam(value, role) {
    const normalized = String(value || "").toLowerCase();
    if (["werewolf", "werewolves", "wolf", "wolves", "狼人", "狼人阵营"].includes(normalized)) return "wolf";
    if (["village", "villagers", "good", "good_people", "好人", "村庄", "好人阵营"].includes(normalized)) return "good";
    return ROLE_META[String(role || "").toLowerCase()]?.team || "unknown";
  }

  function normalizePlayer(player, index) {
    const roleValue = firstDefined(player.role, player.identity, player.known_role, player.character);
    const role = roleValue ? String(roleValue).toLowerCase() : null;
    const id = String(firstDefined(player.id, player.player_id, player.agent_id, player.seat, `p${index + 1}`));
    const aliveValue = firstDefined(player.alive, player.is_alive, player.status !== "dead", true);
    return {
      id,
      name: String(firstDefined(player.name, player.display_name, player.agent_name, `${index + 1}号 Agent`)),
      seat: Number(firstDefined(player.seat, player.seat_no, index + 1)),
      role,
      team: normalizeTeam(firstDefined(player.team, player.faction, player.camp), role),
      alive: aliveValue !== false && String(aliveValue).toLowerCase() !== "dead",
      controller: String(firstDefined(player.controller, player.controlled_by, player.is_human ? "human" : "ai")),
      model: firstDefined(player.model, player.agent_version, player.strategy),
      deathReason: firstDefined(player.deathReason, player.death_reason, player.eliminated_by, null),
      deathRound: firstDefined(player.deathRound, player.death_round, null),
      isSheriff: booleanValue(firstDefined(player.is_sheriff, player.isSheriff), false),
    };
  }

  function sanitizeConfig(config, { includeSeed = false } = {}) {
    if (!config || typeof config !== "object") return {};
    const safeKeys = [
      "player_count", "player_names", "human_player", "human_players",
      "reveal_roles_on_death", "reveal_vote_choices", "shuffle_roles",
      "vote_tie_policy", "tie_rule", "witch_one_potion_per_night",
      "allow_witch_self_save", "hunter_can_shoot_if_poisoned",
      "discussion_rounds", "max_rounds", "strategy", "enable_sheriff",
    ];
    const sanitized = Object.fromEntries(safeKeys
      .filter((key) => Object.prototype.hasOwnProperty.call(config, key))
      .map((key) => [key, Array.isArray(config[key]) ? [...config[key]] : config[key]]));
    if (includeSeed && Object.prototype.hasOwnProperty.call(config, "seed")) sanitized.seed = config.seed;
    return sanitized;
  }

  function normalizePlayerIds(value) {
    if (!value) return [];
    const values = Array.isArray(value) ? value : typeof value === "object" ? Object.keys(value) : [value];
    return [...new Set(values.map((item) => String(firstDefined(item?.player_id, item?.id, item))).filter(Boolean))];
  }

  function normalizeSheriff(value, config = {}) {
    const source = value && typeof value === "object" ? value : {};
    const enabled = booleanValue(firstDefined(source.enabled, source.enable_sheriff, config.enable_sheriff), false);
    const holderId = firstDefined(source.holder_id, source.holder, source.sheriff_id, null);
    return {
      enabled,
      badgeStatus: String(firstDefined(source.badge_status, source.status, holderId ? "holder" : enabled ? "unassigned" : "void")),
      holderId: holderId === null || holderId === undefined ? null : String(holderId),
      candidates: normalizePlayerIds(firstDefined(source.candidates, source.candidate_ids, [])),
      withdrawn: normalizePlayerIds(firstDefined(source.withdrawn, source.withdrawn_ids, [])),
      pkCandidates: normalizePlayerIds(firstDefined(source.pk_candidates, source.pk_candidate_ids, [])),
      voteRound: Number(firstDefined(source.vote_round, source.round, 0)),
      speechOrder: normalizePlayerIds(firstDefined(source.speech_order, source.election_speech_order, [])),
      dayDirection: firstDefined(source.day_direction, source.direction, null),
      daySpeechOrder: normalizePlayerIds(firstDefined(source.day_speech_order, source.current_speech_order, [])),
      nominationTarget: firstDefined(source.nomination_target, source.nomination_target_id, source.sheriff_nomination, null),
    };
  }

  function normalizePendingAction(pending) {
    if (!pending || typeof pending !== "object") return null;
    const legalActions = asArray(firstDefined(pending.legal_actions, pending.actions, [])).map((action) => {
      const targets = firstDefined(action.target_ids, action.targets, []);
      return {
        action_type: String(firstDefined(action.action_type, action.type, "pass")),
        target_ids: Array.isArray(targets) ? [...targets] : targets === null || targets === undefined ? [] : [String(targets)],
        can_pass: Boolean(action.can_pass),
        content_required: Boolean(action.content_required),
        description: String(action.description || ""),
      };
    });
    return {
      actor_id: firstDefined(pending.actor_id, pending.actor, null),
      phase: firstDefined(pending.phase, null),
      round: firstDefined(pending.round, null),
      prompt: String(pending.prompt || ""),
      required: Boolean(firstDefined(pending.required, true)),
      legal_actions: legalActions,
    };
  }

  function cloneProjection(value, fallback) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return fallback;
    }
  }

  function openModal(dialog) {
    if (!dialog) return;
    if (typeof dialog.showModal === "function") {
      if (!dialog.open) dialog.showModal();
      return;
    }
    // The Codex in-app browser and some embedded WebViews expose <dialog>
    // as an element but omit the native modal methods.  Keep the same markup
    // and provide an accessible CSS-backed fallback.
    dialog.setAttribute("open", "");
    dialog.classList.add("dialog-fallback-open");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    document.body.classList.add("dialog-fallback-active");
    requestAnimationFrame(() => dialog.querySelector("button, input, select, textarea")?.focus());
  }

  function closeModal(dialog) {
    if (!dialog) return;
    if (typeof dialog.close === "function") {
      if (dialog.open) dialog.close();
    } else {
      dialog.removeAttribute("open");
      dialog.classList.remove("dialog-fallback-open");
      dialog.removeAttribute("aria-modal");
      if (!document.querySelector("dialog.dialog-fallback-open")) {
        document.body.classList.remove("dialog-fallback-active");
      }
    }
  }

  function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.entries(value).map(([key, item]) => ({ id: key, ...(item || {}) }));
    return [];
  }

  function normalizeEvent(event, index = 0) {
    const type = String(firstDefined(event.event_type, event.type, event.kind, "event")).toLowerCase();
    const data = event.data && typeof event.data === "object" ? event.data : {};
    const visibilityRaw = firstDefined(event.visibility, data.visibility, "public");
    const visibility = Array.isArray(visibilityRaw) ? visibilityRaw.join(",") : String(visibilityRaw);
    const seq = firstDefined(event.seq, event.sequence, event.id, index + 1);
    return {
      ...event,
      seq,
      type,
      round: Number(firstDefined(event.round, data.round, 0)),
      phase: String(firstDefined(event.phase, data.phase, "")),
      actor: firstDefined(event.actor_id, event.actor, data.actor_id, data.actor),
      target: firstDefined(event.target_id, event.target, data.target_id, data.target),
      message: String(firstDefined(event.message, event.content, event.text, data.message, data.content, EVENT_META[type]?.label, type)),
      timestamp: firstDefined(event.timestamp, event.created_at, event.time),
      visibility,
      data,
    };
  }

  function normalizeGame(payload) {
    if (!payload || typeof payload !== "object") return null;
    if (payload.active === false || payload.game === null || payload.state === null && Object.keys(payload).length <= 2) return null;
    const envelope = payload.game || payload.state || payload.current_game || payload.snapshot || payload;
    if (!envelope || typeof envelope !== "object") return null;
    const source = envelope.game_state ? { ...envelope, ...envelope.game_state } : envelope;
    const hasStateFields = ["players", "player_states", "seats", "phase", "current_phase", "status", "game_status", "round", "round_number", "winner", "pending_action", "config", "running"].some((key) => Object.prototype.hasOwnProperty.call(source, key));
    if (!hasStateFields && (payload.event || payload.events || payload.last_seq !== undefined)) return null;
    const id = firstDefined(source.game_id, source.id, envelope.game_id, payload.game_id);
    const playerSource = firstDefined(source.players, source.player_states, source.seats, payload.players, []);
    const pending = firstDefined(source.pending_action, source.pendingAction, payload.pending_action, payload.pendingAction);
    const evaluation = firstDefined(source.evaluation, source.metrics, payload.evaluation, payload.metrics, {});
    const statusRaw = String(firstDefined(source.status, source.game_status, source.finished ? "finished" : "running")).toLowerCase();
    const winner = firstDefined(source.winner, source.winning_team, source.winning_faction, evaluation?.result_metrics?.winner, payload.winner);
    const phase = String(firstDefined(source.stage, source.phase, source.current_phase, payload.phase, statusRaw === "finished" ? "finished" : "setup")).toLowerCase();
    if (!id && !asArray(playerSource).length && !source.phase && !source.status) return null;
    return {
      id: String(id || "local"),
      players: asArray(playerSource).map(normalizePlayer).sort((a, b) => a.seat - b.seat),
      round: Number(firstDefined(source.round, source.round_number, source.day, 0)),
      phase,
      status: statusRaw,
      running: Boolean(firstDefined(source.running, payload.running, false)),
      winner: winner ? String(winner).toLowerCase() : null,
      winReason: firstDefined(source.win_reason, source.winReason, evaluation?.result_metrics?.win_reason, null),
      pendingAction: normalizePendingAction(pending),
      evaluation: cloneProjection(evaluation || {}, {}),
      turningPoints: cloneProjection(firstDefined(source.turning_points, source.turningPoints, evaluation?.turning_points, []), []),
      mode: firstDefined(source.mode, source.game_mode, source.config?.human_player || source.config?.human_players?.length ? "mixed" : "ai"),
      model: firstDefined(source.model, source.strategy, source.agent_model, null),
      config: sanitizeConfig(source.config, { includeSeed: store.perspective === "observer" }),
      sheriff: normalizeSheriff(firstDefined(source.sheriff, source.sheriff_state, payload.sheriff), source.config || {}),
      createdAt: firstDefined(source.created_at, source.started_at, payload.created_at),
      updatedAt: firstDefined(source.updated_at, payload.updated_at),
    };
  }

  function transitionGameSnapshot(game) {
    if (!game) return null;
    return {
      id: game.id,
      players: (game.players || []).map((player) => ({
        id: player.id,
        name: player.name,
        seat: player.seat,
        role: null,
        team: "unknown",
        alive: player.alive,
        controller: player.controller,
        model: player.model,
        deathReason: null,
        deathRound: player.deathRound,
        isSheriff: player.isSheriff,
      })),
      round: game.round,
      phase: game.phase,
      status: game.status,
      running: Boolean(game.running),
      winner: game.winner,
      winReason: game.winReason,
      pendingAction: null,
      evaluation: {},
      turningPoints: [],
      mode: game.mode,
      model: game.model,
      config: sanitizeConfig(game.config, { includeSeed: store.perspective === "observer" }),
      sheriff: cloneProjection(game.sheriff, normalizeSheriff(null, game.config)),
      createdAt: game.createdAt,
      // The referee timestamp advances on private decisions.  Do not retain
      // the last omniscient value while transitioning to a lower privilege.
      updatedAt: store.perspective === "observer" ? game.updatedAt : null,
    };
  }

  function eventKey(event) {
    if (event.seq !== undefined && event.seq !== null) return `seq:${event.seq}`;
    return [event.type, event.round, event.phase, event.actor, event.target, event.message].join("|");
  }

  function mergeEvents(events, { replace = false } = {}) {
    const normalized = asArray(events).map(normalizeEvent);
    if (replace) {
      store.events = [];
      store.eventKeys.clear();
    }
    let added = 0;
    normalized.forEach((event) => {
      const key = eventKey(event);
      if (!store.eventKeys.has(key)) {
        store.eventKeys.add(key);
        store.events.push(event);
        added += 1;
      }
    });
    store.events.sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0));
    if (store.events.length > MAX_EVENTS) {
      store.events = store.events.slice(-MAX_EVENTS);
      store.eventKeys = new Set(store.events.map(eventKey));
    }
    if (added && store.sound) playEventTone();
    return added;
  }

  function extractEvents(payload) {
    if (!payload || typeof payload !== "object") return [];
    if (Array.isArray(payload)) return payload;
    return firstDefined(payload.events, payload.event_log, payload.log, payload.game?.events, payload.state?.events, payload.snapshot?.events, []);
  }

  function applyPayload(payload, { replaceEvents = false } = {}) {
    if (!payload) return;
    const events = extractEvents(payload);
    if (events.length) mergeEvents(events, { replace: replaceEvents });
    if (payload.event && typeof payload.event === "object") mergeEvents([payload.event]);
    const nextGame = normalizeGame(payload);
    if (nextGame) {
      const previousId = store.game?.id;
      if (previousId && previousId !== nextGame.id) {
        store.events = [];
        store.eventKeys.clear();
        mergeEvents(events, { replace: true });
        store.resultShownFor = null;
      }
      store.game = nextGame;
      store.auto = nextGame.running && !isFinished(nextGame);
    }
    renderAll();
  }

  async function request(path, options = {}) {
    const timeout = options.timeout ?? 9000;
    const controller = options.signal ? null : new AbortController();
    const timer = controller ? window.setTimeout(() => controller.abort(), timeout) : null;
    const headers = { Accept: "application/json", ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
    try {
      const response = await fetch(`${API_BASE}${path}`, { ...options, headers, signal: options.signal || controller.signal });
      const raw = await response.text();
      let data = null;
      if (raw) {
        try { data = JSON.parse(raw); } catch { data = { message: raw }; }
      }
      if (!response.ok) {
        const message = firstDefined(data?.detail, data?.message, data?.error, `${response.status} ${response.statusText}`);
        throw new Error(String(message));
      }
      return data;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  function applyProviderCapabilities(payload = {}) {
    const capability = payload?.providers?.deepseek || {};
    const configured = capability.configured === true;
    const model = String(capability.model || "deepseek-flash");
    store.providers.deepseek = { configured, model };

    const select = $("#strategy-select");
    const option = $("#deepseek-option");
    if (!select || !option) return;
    option.disabled = !configured;
    option.textContent = configured
      ? `DeepSeek · ${model}`
      : "DeepSeek · 服务器未配置";
    if (!select.dataset.userSelected) {
      select.value = configured ? "deepseek" : "heuristic";
    } else if (!configured && select.value === "deepseek") {
      select.value = "heuristic";
    }
  }

  async function loadCapabilities() {
    const health = await request("/api/health", { timeout: 5000 });
    applyProviderCapabilities(health);
    return health;
  }

  function statePath() {
    const params = new URLSearchParams();
    if (store.game?.id) params.set("game_id", store.game.id);
    appendPerspective(params);
    return `/api/state${params.size ? `?${params}` : ""}`;
  }

  function eventsPath(gameId, after = 0) {
    const params = new URLSearchParams({ after: String(after) });
    appendPerspective(params);
    return `/api/games/${encodeURIComponent(gameId)}/events?${params}`;
  }

  function appendPerspective(params) {
    if (store.perspective === "observer") {
      params.set("omniscient", "true");
    } else if (store.perspective === "public") {
      params.set("omniscient", "false");
    } else if (store.perspective) {
      params.set("viewer", store.perspective);
      params.set("omniscient", "false");
    }
    return params;
  }

  async function loadState({ quiet = false } = {}) {
    const revision = store.viewRevision;
    if (!quiet) setConnection("connecting", "正在连接", "检查对局服务…");
    try {
      const data = await request(statePath());
      if (revision !== store.viewRevision) return null;
      applyPayload(data, { replaceEvents: !store.events.length });
      setConnection("connected", "服务已连接", store.socket?.readyState === WebSocket.OPEN ? "WebSocket 实时流" : "HTTP 状态同步");
      if (store.game?.id) {
        connectSocket(store.game.id);
        startPolling();
      }
      return data;
    } catch (error) {
      if (!quiet) {
        setConnection("error", "服务未连接", API_BASE || "当前来源");
        toast("无法连接对局服务", humanizeError(error), "error");
      }
      throw error;
    }
  }

  function setConnection(kind, label, detail) {
    store.connection = kind;
    const indicator = $("#connection-indicator");
    indicator.className = `connection-indicator ${kind}`;
    $("#connection-label").textContent = label;
    $("#connection-detail").textContent = detail;
  }

  function connectSocket(gameId) {
    if (!gameId || typeof WebSocket === "undefined") return;
    if (store.socket && store.socket.__gameId === gameId && [WebSocket.OPEN, WebSocket.CONNECTING].includes(store.socket.readyState)) return;
    closeSocket();
    const base = API_BASE || window.location.origin;
    let url;
    try {
      const parsed = new URL(base);
      parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
      parsed.pathname = `/ws/${encodeURIComponent(gameId)}`;
      const socketParams = new URLSearchParams();
      appendPerspective(socketParams);
      parsed.search = socketParams.toString();
      url = parsed.toString();
    } catch {
      return;
    }
    try {
      const revision = store.viewRevision;
      const socket = new WebSocket(url);
      socket.__gameId = gameId;
      store.socket = socket;
      socket.addEventListener("open", () => setConnection("connected", "实时流已连接", `对局 ${shortId(gameId)}`));
      socket.addEventListener("message", (message) => {
        if (revision !== store.viewRevision) return;
        try {
          const data = JSON.parse(message.data);
          applyPayload(data);
        } catch {
          // Malformed socket messages are ignored without breaking polling.
        }
      });
      socket.addEventListener("close", () => {
        if (store.socket !== socket) return;
        setConnection("connected", "服务已连接", "实时流中断，已切换轮询");
        if (store.game?.id === gameId && !isFinished(store.game)) {
          clearTimeout(store.reconnectTimer);
          store.reconnectTimer = window.setTimeout(() => connectSocket(gameId), 3500);
        }
      });
      socket.addEventListener("error", () => socket.close());
    } catch {
      setConnection("connected", "服务已连接", "使用 HTTP 轮询");
    }
  }

  function closeSocket() {
    clearTimeout(store.reconnectTimer);
    if (store.socket) {
      store.socket.onclose = null;
      store.socket.close();
      store.socket = null;
    }
  }

  function startPolling() {
    clearInterval(store.pollTimer);
    if (!store.game?.id) return;
    store.pollTimer = window.setInterval(async () => {
      const revision = store.viewRevision;
      const id = store.game?.id;
      if (!id || document.hidden) return;
      try {
        const after = store.events.at(-1)?.seq ?? 0;
        const data = await request(eventsPath(id, after), { timeout: 5000 });
        if (revision !== store.viewRevision) return;
        const added = mergeEvents(extractEvents(data).length ? extractEvents(data) : Array.isArray(data) ? data : []);
        if (added) renderAll();
        if (store.socket?.readyState !== WebSocket.OPEN || store.auto) await loadState({ quiet: true });
      } catch {
        if (store.socket?.readyState !== WebSocket.OPEN) setConnection("error", "连接不稳定", "正在自动重试");
      }
    }, 2200);
  }

  async function createGame(form) {
    const values = new FormData(form);
    const mode = values.get("mode") || "ai";
    const seedValue = String(values.get("seed") || "").trim();
    const seatValue = values.get("human_seat");
    const humanName = String(values.get("human_name") || "玩家").trim();
    const playerCount = Number(values.get("player_count") || 8);
    const humanSeat = seatValue === "random" ? 1 : Number(seatValue || 1);
    const playerNames = Array.from({ length: playerCount }, (_, index) => mode === "mixed" && index + 1 === humanSeat ? humanName : `Agent ${String(index + 1).padStart(2, "0")}`);
    const tieRule = values.get("tie_rule") === "random" ? "random" : "no_elimination";
    const payload = {
      mode,
      player_count: playerCount,
      player_names: playerNames,
      seed: seedValue ? Number(seedValue) : Math.floor(Math.random() * 1_000_000),
      strategy: values.get("strategy"),
      vote_tie_policy: tieRule,
      enable_sheriff: values.get("enable_sheriff") === "on",
      human_player: mode === "mixed" ? humanSeat : null,
      human_players: mode === "mixed" ? [humanSeat] : [],
    };

    const submit = $("#create-submit");
    submit.disabled = true;
    submit.innerHTML = "<span>◌</span> 正在创建";
    try {
      const data = await request("/api/games", { method: "POST", body: JSON.stringify(payload), timeout: 15000 });
      store.events = [];
      store.eventKeys.clear();
      store.resultShownFor = null;
      if (mode === "mixed") {
        const created = normalizeGame(data);
        if (created) {
          const human = created.players.find((player) => player.controller === "human") || created.players.find((player) => player.seat === humanSeat);
          store.perspective = human?.id || `p${humanSeat}`;
          store.viewRevision += 1;
          store.game = transitionGameSnapshot(created);
          renderAll();
          try {
            await loadState({ quiet: true });
          } catch (error) {
            toast("对局已创建，正在重试玩家视角", humanizeError(error), "info");
          }
        } else {
          applyPayload(data, { replaceEvents: true });
        }
      } else {
        if (store.perspective !== "observer") store.viewRevision += 1;
        store.perspective = "observer";
        applyPayload(data, { replaceEvents: true });
      }
      closeModal($("#create-dialog"));
      navigate("arena");
      if (store.game?.id) {
        connectSocket(store.game.id);
        startPolling();
      }
      setConnection("connected", "服务已连接", store.game ? `对局 ${shortId(store.game.id)}` : "新对局已创建");
      toast("对局已创建", `${playerCount} 名玩家已进入 Agent 对局席。`, "success");
    } catch (error) {
      toast("创建失败", humanizeError(error), "error");
    } finally {
      submit.disabled = false;
      submit.innerHTML = "<span>✦</span> 创建并进入";
    }
  }

  async function stepGame(action = null) {
    const game = store.game;
    if (!game || isFinished(game)) return;
    const gameId = game.id;
    const revision = store.viewRevision;
    const perspective = store.perspective;
    const button = $("#step-game");
    button.disabled = true;
    try {
      const body = action ? { action, ...action } : {};
      body.viewer = perspective;
      body.omniscient = perspective === "observer";
      const data = await request(`/api/games/${encodeURIComponent(gameId)}/step`, { method: "POST", body: JSON.stringify(body), timeout: 20000 });
      if (revision !== store.viewRevision || store.game?.id !== gameId) return;
      applyPayload(data);
      await pollEvents({ quiet: true });
    } catch (error) {
      if (revision !== store.viewRevision || store.game?.id !== gameId) return;
      toast("推进失败", humanizeError(error), "error");
    } finally {
      renderControls();
      renderHumanTurn();
    }
  }

  async function runGame() {
    const game = store.game;
    if (!game || isFinished(game)) return;
    const gameId = game.id;
    const revision = store.viewRevision;
    const perspective = store.perspective;
    if (store.auto) {
      store.runController?.abort();
      store.runController = null;
      try {
        const data = await request(`/api/games/${encodeURIComponent(gameId)}/stop`, {
          method: "POST",
          body: JSON.stringify({ viewer: perspective, omniscient: perspective === "observer" }),
          timeout: 6000,
        });
        if (revision !== store.viewRevision || store.game?.id !== gameId) return;
        applyPayload(data);
        toast("自动运行已停止", "你可以单步推进或稍后继续。", "info");
      } catch {
        if (revision !== store.viewRevision || store.game?.id !== gameId) return;
        store.auto = false;
        if (store.game) store.game.running = false;
        renderControls();
        toast("已停止跟随", "后端可能不支持暂停，但界面已停止自动跟随。", "info");
      }
      return;
    }
    store.auto = true;
    const controller = new AbortController();
    store.runController = controller;
    renderControls();
    const speed = Number($("#speed-select").value || 850);
    try {
      const data = await request(`/api/games/${encodeURIComponent(gameId)}/run`, {
        method: "POST",
        body: JSON.stringify({ delay_ms: speed, step_delay_ms: speed, viewer: perspective, omniscient: perspective === "observer" }),
        signal: controller.signal,
        timeout: 180000,
      });
      if (revision !== store.viewRevision || store.game?.id !== gameId) return;
      applyPayload(data);
      await pollEvents({ quiet: true });
    } catch (error) {
      if (revision !== store.viewRevision || store.game?.id !== gameId) return;
      if (error.name !== "AbortError") {
        store.auto = false;
        toast("自动运行未完成", humanizeError(error), "error");
      }
    } finally {
      if (store.runController === controller) store.runController = null;
      if (revision === store.viewRevision) renderControls();
    }
  }

  async function pollEvents({ quiet = false } = {}) {
    if (!store.game?.id) return;
    const revision = store.viewRevision;
    try {
      const after = store.events.at(-1)?.seq ?? 0;
      const data = await request(eventsPath(store.game.id, after));
      if (revision !== store.viewRevision) return;
      const events = extractEvents(data).length ? extractEvents(data) : Array.isArray(data) ? data : [];
      if (mergeEvents(events)) renderAll();
      await loadState({ quiet: true });
    } catch (error) {
      if (!quiet) toast("事件同步失败", humanizeError(error), "error");
    }
  }

  function isFinished(game) {
    return !game || game.status === "finished" || game.phase === "finished" || Boolean(game.winner);
  }

  function shortId(id) {
    const value = String(id || "");
    return value.length > 12 ? `${value.slice(0, 8)}…` : value;
  }

  function humanizeError(error) {
    if (error?.name === "AbortError") return "请求已取消或超时。";
    const message = String(error?.message || error || "未知错误");
    if (/failed to fetch|networkerror/i.test(message)) return `无法访问 ${API_BASE || "同源后端"}，请确认服务已启动。`;
    return message;
  }

  function phaseInfo(phase) {
    const key = String(phase || "setup").toLowerCase();
    if (PHASE_META[key]) return PHASE_META[key];
    if (key.includes("night")) return PHASE_META.night;
    if (key.includes("sheriff") || key.includes("badge")) return PHASE_META.sheriff_run;
    if (key.includes("dawn") || key.includes("reveal")) return PHASE_META.dawn_reveal;
    if (key.includes("pk") && key.includes("speak")) return PHASE_META.day_pk_speech;
    if (key.includes("discuss") || key.includes("speak")) return PHASE_META.day_discussion;
    if (key.includes("vote")) return PHASE_META.day_vote;
    if (key.includes("finish") || key.includes("game_over")) return PHASE_META.finished;
    return { ...PHASE_META.setup, name: key || "准备" };
  }

  function winnerName(winner) {
    const value = String(winner || "").toLowerCase();
    if (["village", "villagers", "good", "good_people", "好人", "村庄"].includes(value)) return "好人阵营";
    if (["werewolf", "werewolves", "wolf", "wolves", "狼人"].includes(value)) return "狼人阵营";
    if (value === "draw") return "平局";
    return winner ? String(winner) : "胜负未定";
  }

  function roleInfo(role) {
    return ROLE_META[String(role || "unknown").toLowerCase()] || { name: String(role || "身份未知"), icon: "?", team: "unknown" };
  }

  function playerName(id) {
    if (!id) return "系统";
    return store.game?.players.find((player) => String(player.id) === String(id))?.name || String(id);
  }

  function eventInfo(type) {
    const key = String(type || "event").toLowerCase();
    if (EVENT_META[key]) return EVENT_META[key];
    if (key.includes("sheriff") || key.includes("badge")) return { icon: "♛", label: "警长事件", category: "sheriff" };
    const match = Object.keys(EVENT_META).find((candidate) => key.includes(candidate));
    return match ? EVENT_META[match] : { icon: "·", label: key.replaceAll("_", " "), category: "action" };
  }

  function displayEventMessage(event) {
    const data = event.data || {};
    const actor = playerName(event.actor);
    const target = event.target ? playerName(event.target) : null;
    const type = event.type;
    if (["game_started", "game_start"].includes(type)) return `对局开始，共 ${firstDefined(data.player_count, store.game?.players.length, "多")} 名玩家入场。`;
    if (type === "role_assigned") return `${actor} 已收到私有身份${data.role ? `：${roleInfo(data.role).name}` : ""}。`;
    if (type === "wolf_team") return `${actor} 已获得狼队同伴信息。`;
    if (["phase_started", "phase_changed", "phase_change"].includes(type)) return `裁判进入「${phaseInfo(firstDefined(data.phase, event.phase)).name}」阶段。`;
    if (type === "night_started") return `第 ${event.round || store.game?.round || 1} 夜开始，所有玩家闭眼。`;
    if (type === "dawn_reveal_started") return `第 ${event.round || store.game?.round || 1} 天黎明，法官开始公布昨夜结果。`;
    if (type === "day_started") return `第 ${event.round || store.game?.round || 1} 天讨论开始。`;
    if (type === "vote_started") return `第 ${event.round || store.game?.round || 1} 天放逐投票开始。`;
    if (type === "sheriff_election_started") return "首日警长竞选开始，所有玩家正在同时决定是否上警。";
    if (type === "sheriff_run_choice") return `${actor} 的上警选择已封存：${data.running ? "上警" : "不上警"}。`;
    if (type === "sheriff_candidacy_result") {
      const candidates = playerList(firstDefined(data.candidates, data.candidate_ids, []));
      return candidates ? `上警玩家：${candidates}。` : "无人上警，本局警徽流失。";
    }
    if (type === "sheriff_speech") return event.message || `${actor} 完成警上发言。`;
    if (type === "sheriff_withdrawal_result") {
      const withdrawn = playerList(firstDefined(data.withdrawn, data.withdrawn_ids, []));
      const remaining = playerList(firstDefined(data.candidates, data.remaining_candidates, []));
      return `${withdrawn ? `退水玩家：${withdrawn}` : "无人退水"}${remaining ? `；留在警上：${remaining}` : ""}。`;
    }
    if (type === "sheriff_withdraw_choice") return `${actor} 的退水选择已封存：${data.withdrew ? "退水" : "继续竞选"}。`;
    if (type === "sheriff_vote_started") return `${firstDefined(data.pk, data.is_pk, Number(data.round) > 1, false) ? "警长 PK 投票" : "警长投票"}开始，原警下玩家拥有投票权。`;
    if (type === "sheriff_vote_cast") return data.abstained ? `${actor} 的弃票已封存。` : `${actor} 的警长票已封存：${playerName(event.target)}。`;
    if (type === "sheriff_vote_result") return `${firstDefined(data.pk, data.is_pk, Number(data.round) > 1, false) ? "PK " : ""}警长票型：${formatVoteTally(firstDefined(data.tally, data.vote_tally, data.totals, data.votes)) || "无人投票"}。`;
    if (type === "sheriff_pk_started") return `警长竞选出现平票，${playerList(firstDefined(data.candidates, data.pk_candidates, [])) || "平票候选人"}进入 PK。`;
    if (type === "sheriff_elected") {
      const elected = playerName(firstDefined(event.target, data.holder_id, data.sheriff_id, data.elected_id));
      return `${elected} 获得警徽，成为本局警长。`;
    }
    if (type === "sheriff_election_void") return `警长竞选${data.reason ? `因${String(data.reason)}` : ""}流局，本局此后不再有警长。`;
    if (type === "sheriff_order") {
      const order = playerList(firstDefined(data.speech_order, data.day_speech_order, data.order, []));
      const direction = directionLabel(firstDefined(data.direction, data.day_direction));
      return `${actor} 选择${direction || "发言方向"}${order ? `，顺序为 ${order}` : ""}。`;
    }
    if (type === "sheriff_summary") {
      const nomination = firstDefined(event.target, data.nomination_target, data.target_id);
      const suffix = nomination ? `（归票：${playerName(nomination)}）` : "";
      return `${event.message || `${actor} 完成警长总结`}${suffix}`;
    }
    if (type === "badge_transfer_required") return `${actor} 已出局，正在决定移交或撕毁警徽。`;
    if (type === "badge_transferred") {
      const nextHolder = playerName(firstDefined(event.target, data.to, data.new_holder_id, data.holder_id));
      return `${actor} 将警徽移交给 ${nextHolder}。`;
    }
    if (type === "badge_destroyed") return `${actor} 选择撕毁警徽，本局此后不再有警长。`;
    if (type === "day_vote_result") return `放逐票型：${formatVoteTally(firstDefined(data.tally, data.vote_tally, data.totals, data.votes)) || "暂无有效票"}。`;
    if (type === "day_pk_started") return `放逐投票出现平票，${playerList(firstDefined(data.candidates, data.pk_candidates, [])) || "平票玩家"}进入 PK。`;
    if (type === "day_pk_speech") return event.message || `${actor} 完成放逐 PK 发言。`;
    if (type === "hunter_shot") return `${actor}${target ? ` 向 ${target} 开枪` : " 已执行猎人技能"}。`;
    if (["wolf_kill", "wolf_vote"].includes(type)) return `${actor}${target ? ` 将袭击意向指向 ${target}` : " 已提交袭击意向"}。`;
    if (["seer_check", "check"].includes(type)) return `${actor}${target ? ` 查验了 ${target}` : " 完成查验"}${data.result ? `，结果为${normalizeTeam(data.result) === "wolf" ? "狼人阵营" : "好人阵营"}` : ""}。`;
    if (["vote", "vote_cast"].includes(type)) return `${actor}${target ? ` 投票给 ${target}` : " 已提交投票"}。`;
    if (["death", "eliminated"].includes(type) && target) return `${target} 已出局${data.cause ? `（${causeLabel(data.cause)}）` : ""}。`;
    if (type === "night_result") {
      const victims = asArray(firstDefined(data.casualties, data.deaths, [])).map((item) => playerName(item.id || item)).join("、");
      return victims ? `昨夜出局：${victims}。` : "昨夜平安无事。";
    }
    if (["game_finished", "game_over", "game_end"].includes(type)) return `${winnerName(firstDefined(data.winner, data.winning_team, store.game?.winner))}获胜，对局完成。`;
    return event.message;
  }

  function playerList(value) {
    return normalizePlayerIds(value).map(playerName).join("、");
  }

  function directionLabel(value) {
    const key = String(value || "").toLowerCase();
    if (["clockwise", "cw", "顺时针"].includes(key)) return "顺时针发言";
    if (["counterclockwise", "counter_clockwise", "ccw", "逆时针"].includes(key)) return "逆时针发言";
    return value ? String(value) : "";
  }

  function formatVoteTally(value) {
    if (!value) return "";
    if (Array.isArray(value)) {
      return value.map((item) => {
        if (item && typeof item === "object") {
          const id = firstDefined(item.target_id, item.player_id, item.id, item.target);
          const votes = firstDefined(item.weighted_votes, item.weight, item.votes, item.count, item.total, 0);
          return `${playerName(id)} ${Number(votes) % 1 ? votes : Number(votes) > 3 ? Number(votes) / 2 : votes} 票`;
        }
        return playerName(item);
      }).join("、");
    }
    if (typeof value === "object") {
      return Object.entries(value).map(([id, votes]) => `${playerName(id)} ${Number(votes)} 票`).join("、");
    }
    return String(value);
  }

  function causeLabel(cause) {
    const labels = { night: "夜间死亡", werewolf_attack: "狼人袭击", execution: "投票放逐", vote: "投票放逐", poison: "女巫毒杀", hunter: "猎人开枪" };
    return labels[String(cause)] || String(cause);
  }

  function renderAll() {
    renderHeader();
    renderPhaseTrack();
    renderSeats();
    renderEvents();
    renderConversations();
    renderMetrics();
    renderHumanTurn();
    renderControls();
    renderMiniGame();
    renderReplayList();
    maybeShowResult();
  }

  function renderHeader() {
    const game = store.game;
    const phase = phaseInfo(game?.phase);
    $("#phase-heading").textContent = game ? phase.title : "夜幕尚未降临";
    $("#phase-subtitle").textContent = game ? phase.subtitle : "创建一场对局，观察不同角色 Agent 在信息隔离下推理与博弈。";
    $("#round-number").textContent = String(game?.round || 0).padStart(2, "0");
    $("#phase-label").textContent = game ? phase.name : "准备";
    $("#match-id").textContent = game ? `MATCH · ${shortId(game.id)}` : "—";
    const status = $("#match-status");
    const statusKind = game ? isFinished(game) ? "finished" : game.status === "waiting_for_human" ? "running" : "running" : "waiting";
    status.className = `status-pill ${statusKind}`;
    status.innerHTML = `<i></i> ${game ? isFinished(game) ? "已结束" : game.status === "waiting_for_human" ? "等待玩家" : "实时进行" : "等待对局"}`;
    $("#nav-live-dot").hidden = !game || isFinished(game);
    $("#center-kicker").textContent = game ? `ROUND ${String(game.round || 0).padStart(2, "0")}` : "WAITING";
    $("#center-title").textContent = game ? phase.name : "等待开局";
    $("#center-hint").textContent = game ? `${game.players.filter((p) => p.alive).length} / ${game.players.length} ALIVE` : "No active match";
    $("#phase-emblem").textContent = phase.icon;
    renderCenterSheriff();
    populatePerspectives();
  }

  function renderCenterSheriff() {
    const element = $("#center-sheriff");
    const sheriff = store.game?.sheriff;
    if (!sheriff?.enabled) {
      element.hidden = true;
      element.innerHTML = "";
      return;
    }
    const holder = sheriff.holderId ? playerName(sheriff.holderId) : null;
    const statusLabels = { destroyed: "警徽已撕毁", void: "本局无警长", unassigned: "警徽待选出", holder: holder ? `警长 · ${holder}` : "警长已产生" };
    const phase = String(store.game?.phase || "");
    const showDayOrder = [
      "day_discussion", "day_vote", "day_pk_speech", "day_pk_vote", "hunter_shot",
    ].includes(phase);
    const order = showDayOrder ? sheriff.daySpeechOrder.map(playerName) : [];
    const direction = showDayOrder ? directionLabel(sheriff.dayDirection) : "";
    element.hidden = false;
    element.innerHTML = `<span>♛ ${escapeHtml(statusLabels[sheriff.badgeStatus] || statusLabels.unassigned)}</span>${order.length ? `<small>${direction ? `${escapeHtml(direction)} · ` : ""}${escapeHtml(order.join(" → "))}</small>` : ""}`;
  }

  function populatePerspectives() {
    const select = $("#perspective-select");
    const current = store.perspective;
    const base = [
      { value: "observer", label: "全知观战" },
      { value: "public", label: "公共视角" },
    ];
    const players = (store.game?.players || []).map((player) => ({ value: player.id, label: `${player.seat}号 · ${player.name}` }));
    const options = [...base, ...players];
    const signature = options.map((item) => item.value).join("|");
    if (select.dataset.signature !== signature) {
      select.innerHTML = options.map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`).join("");
      select.dataset.signature = signature;
    }
    select.value = options.some((item) => String(item.value) === String(current)) ? current : "observer";
  }

  function renderPhaseTrack() {
    const track = phaseInfo(store.game?.phase).track;
    const nodes = $$(".phase-node", $("#phase-track"));
    nodes.forEach((node, index) => {
      const skipped = Boolean(store.game) && node.dataset.phase === "sheriff" && !store.game.sheriff?.enabled;
      node.classList.toggle("skipped", skipped);
      node.classList.toggle("done", !skipped && Boolean(store.game) && (index < track || track >= nodes.length));
      node.classList.toggle("active", !skipped && (!store.game ? index === 0 : index === Math.min(track, nodes.length - 1) && track < nodes.length));
      const icon = $("i", node);
      icon.textContent = skipped ? "–" : node.classList.contains("done") ? "✓" : String(index + 1);
    });
    $$(".phase-line", $("#phase-track")).forEach((line, index) => line.classList.toggle("done", Boolean(store.game) && index < track));
  }

  function renderSeats() {
    const ring = $("#seat-ring");
    $$(".player-seat", ring).forEach((node) => node.remove());
    const players = store.game?.players || [];
    if (!players.length) return;
    const lastEvent = store.events.at(-1);
    const latestSpeech = [...store.events].reverse().find(isTranscriptEvent);
    const sheriff = store.game?.sheriff || normalizeSheriff(null, store.game?.config);
    const phase = String(store.game?.phase || "");
    const electionActive = [
      "sheriff_run", "sheriff_speech", "sheriff_withdraw", "sheriff_vote", "sheriff_pk_speech", "sheriff_pk_vote",
    ].includes(phase);
    const sheriffPkActive = phase === "sheriff_pk_speech" || phase === "sheriff_pk_vote";
    const dayPkActive = phase === "day_pk_speech" || phase === "day_pk_vote";
    // sheriff.pk_candidates is intentionally election-only.  Exile-PK
    // candidates are recovered from the public day_pk_started event so the UI
    // never revives a historical sheriff runoff as the current exile runoff.
    const dayPkEvent = dayPkActive
      ? [...store.events].reverse().find((event) => event.type === "day_pk_started" && Number(event.round) === Number(store.game?.round))
      : null;
    const activePkCandidates = sheriffPkActive
      ? sheriff.pkCandidates
      : dayPkActive
        ? normalizePlayerIds(firstDefined(dayPkEvent?.data?.candidates, dayPkEvent?.data?.pk_candidates, []))
        : [];
    players.forEach((player, index) => {
      const angle = -Math.PI / 2 + (Math.PI * 2 * index) / players.length;
      const radiusX = players.length > 9 ? 43 : 40;
      const radiusY = players.length > 9 ? 41 : 39;
      const x = 50 + Math.cos(angle) * radiusX;
      const y = 50 + Math.sin(angle) * radiusY;
      const knownRole = player.role && player.role !== "unknown";
      const role = roleInfo(knownRole ? player.role : "unknown");
      let label = role.name;
      let icon = role.icon;
      let teamClass = knownRole ? player.team : "unknown";
      if (store.overlay === "team") {
        label = player.team === "wolf" ? "狼人阵营" : player.team === "good" ? "好人阵营" : "阵营未知";
        icon = player.team === "wolf" ? "♞" : player.team === "good" ? "◇" : "?";
      } else if (store.overlay === "alive") {
        label = player.alive ? "存活" : `已出局${player.deathReason ? ` · ${player.deathReason}` : ""}`;
        icon = player.alive ? "●" : "†";
        teamClass = player.alive ? "good" : "unknown";
      }
      const isSheriff = player.isSheriff || sheriff.holderId === player.id;
      // Candidate/withdrawal/runoff arrays remain in the replayable public
      // state after their phase ends.  Only render them as live seat markers
      // while the corresponding election or runoff is actually in progress.
      const isCandidate = electionActive && sheriff.candidates.includes(player.id) && !sheriff.withdrawn.includes(player.id);
      const isWithdrawn = electionActive && sheriff.withdrawn.includes(player.id);
      const isPk = activePkCandidates.includes(player.id);
      const pkLabel = sheriffPkActive ? "警长竞选 PK" : "放逐 PK";
      const isNominated = sheriff.nominationTarget !== null && String(sheriff.nominationTarget) === player.id;
      const markers = [
        isSheriff ? `<span class="seat-marker badge" title="警长">♛</span>` : "",
        isPk ? `<span class="seat-marker pk">PK</span>` : isWithdrawn ? `<span class="seat-marker withdrawn">退水</span>` : isCandidate ? `<span class="seat-marker candidate">警上</span>` : "",
        isNominated ? `<span class="seat-marker nominated">归票</span>` : "",
      ].filter(Boolean).join("");
      const seat = document.createElement("article");
      seat.className = `player-seat ${teamClass}${player.alive ? "" : " dead"}${latestSpeech?.actor && String(latestSpeech.actor) === player.id ? " speaking" : ""}${lastEvent?.target && String(lastEvent.target) === player.id ? " targeted" : ""}${isSheriff ? " sheriff-holder" : ""}${isCandidate ? " sheriff-candidate" : ""}${isWithdrawn ? " sheriff-withdrawn" : ""}${isPk ? " sheriff-pk" : ""}${isNominated ? " sheriff-nominated" : ""}`;
      seat.style.setProperty("--seat-x", `${x}%`);
      seat.style.setProperty("--seat-y", `${y}%`);
      seat.title = `${player.seat}号 ${player.name} · ${label}${isSheriff ? " · 警长" : isPk ? ` · ${pkLabel}` : isWithdrawn ? " · 已退水" : isCandidate ? " · 警上候选" : ""}`;
      seat.innerHTML = `
        <span class="seat-number">${escapeHtml(player.seat)}</span>
        <div class="seat-avatar">${escapeHtml(icon)}</div>
        <div class="seat-copy"><strong>${escapeHtml(player.name)}</strong><span>${escapeHtml(label)}</span></div>
        <div class="seat-markers">${markers}</div>
        <span class="seat-badge ${player.controller === "human" ? "human" : ""}">${player.controller === "human" ? "HUMAN" : escapeHtml(player.model || "AGENT")}</span>`;
      ring.appendChild(seat);
    });
  }

  function renderEvents() {
    const stream = $("#event-stream");
    $("#event-count").textContent = String(store.events.length);
    const filtered = store.events.filter((event) => {
      if (store.eventFilter === "all") return true;
      const category = eventInfo(event.type).category;
      if (store.eventFilter === "action") return ["action", "danger", "system"].includes(category);
      return category === store.eventFilter;
    });
    if (!filtered.length) {
      stream.innerHTML = `<div class="empty-state compact-empty"><span class="empty-glyph">⌁</span><strong>${store.events.length ? "没有符合筛选的事件" : "等待第一条事件"}</strong><p>${store.events.length ? "选择其他类型以查看事件。" : "Agent 的行动、发言和规则裁决会依次出现在这里。"}</p></div>`;
      return;
    }
    stream.innerHTML = filtered.slice(-150).map((event) => {
      const meta = eventInfo(event.type);
      const actor = event.actor ? playerName(event.actor) : meta.label;
      const visibility = event.visibility && event.visibility !== "public" ? `<span class="visibility-tag">${escapeHtml(visibilityLabel(event.visibility))}</span>` : "";
      return `<article class="event-item ${meta.category}">
        <span class="event-symbol">${escapeHtml(meta.icon)}</span>
        <strong>${escapeHtml(actor)}</strong>
        <time>${escapeHtml(formatEventTime(event))}</time>
        <p>${escapeHtml(displayEventMessage(event))}${visibility}</p>
      </article>`;
    }).join("");
    requestAnimationFrame(() => { stream.scrollTop = stream.scrollHeight; });
  }

  function visibilityLabel(value) {
    if (String(value).includes("observer")) return "仅观战可见";
    if (String(value).includes("wolves")) return "狼队私有";
    return "私有事件";
  }

  function formatEventTime(event) {
    if (event.timestamp) {
      const date = new Date(event.timestamp);
      if (!Number.isNaN(date.valueOf())) return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
    }
    return event.round ? `R${event.round} · #${event.seq}` : `#${event.seq}`;
  }

  function renderConversations() {
    const list = $("#conversation-list");
    const speeches = store.events.filter(isTranscriptEvent);
    if (!speeches.length) {
      list.innerHTML = `<div class="empty-state inline-empty"><span class="empty-glyph">“</span><div><strong>尚无公开发言</strong><p>进入白天讨论后，Agent 的发言会在此展开。</p></div></div>`;
      return;
    }
    list.innerHTML = speeches.slice(-60).map((event) => {
      const player = store.game?.players.find((item) => String(item.id) === String(event.actor));
      const role = roleInfo(player?.role);
      const context = speechContext(event);
      const metadata = firstDefined(event.metadata, event.data?.metadata, {});
      const badgeFlow = firstDefined(metadata?.badge_flow, event.data?.badge_flow, event.badge_flow);
      const nomination = firstDefined(metadata?.sheriff_nomination, event.data?.sheriff_nomination, event.data?.nomination_target);
      return `<article class="speech-row">
        <div class="speech-avatar">${escapeHtml(role.icon)}</div>
        <div><div class="speech-meta"><strong>${escapeHtml(player?.name || event.actor || "未知 Agent")}</strong><span>${escapeHtml(player?.role ? role.name : "公开频道")}</span>${context ? `<span class="speech-context">${escapeHtml(context)}</span>` : ""}<time>${escapeHtml(formatEventTime(event))}</time></div><p class="speech-bubble">${escapeHtml(displayEventMessage(event))}</p>${renderBadgeFlow(badgeFlow)}${nomination ? `<div class="nomination-chip">♛ 归票建议 · ${escapeHtml(playerName(nomination))}</div>` : ""}</div>
      </article>`;
    }).join("");
    requestAnimationFrame(() => { list.scrollTop = list.scrollHeight; });
  }

  function isTranscriptEvent(event) {
    const type = String(event?.type || "");
    return eventInfo(type).category === "speech"
      || ["speak", "speech", "statement", "day_speech", "sheriff_speech", "sheriff_pk_speech", "sheriff_summary", "day_pk_speech"].some((key) => type.includes(key));
  }

  function speechContext(event) {
    const type = String(event?.type || "");
    const phase = String(event?.phase || "");
    if (type.includes("sheriff_summary")) return "警长总结 · 归票";
    if (type.includes("sheriff_speech") || phase.includes("sheriff_speech")) return "警上发言";
    if (type.includes("day_pk_speech") || phase.includes("day_pk_speech")) return "放逐 PK";
    if (phase.includes("sheriff_pk_speech")) return "警长竞选 PK";
    return "";
  }

  function renderBadgeFlow(flow) {
    if (!flow || typeof flow !== "object") return "";
    const targets = normalizePlayerIds(firstDefined(flow.targets, flow.target_ids, [])).slice(0, 2);
    if (!targets.length) return "";
    const mode = String(firstDefined(flow.mode, targets.length > 1 ? "double" : "single")).toLowerCase();
    const fallback = firstDefined(flow.fallback_target, flow.fallback_target_id, null);
    return `<div class="badge-flow-card"><div><span>♛</span><strong>${mode === "double" ? "双夜警徽流" : "单夜警徽流"}</strong></div><p>${targets.map((id, index) => `第${index + 1}验 ${playerName(id)}`).map(escapeHtml).join(" · ")}${fallback ? `<br>外置位：${escapeHtml(playerName(fallback))}` : ""}</p></div>`;
  }

  function metricValue(evaluation, ...paths) {
    for (const path of paths) {
      const value = path.split(".").reduce((current, key) => current?.[key], evaluation);
      const normalized = percent(value);
      if (normalized !== null) return normalized;
    }
    return null;
  }

  function renderMetrics() {
    const evaluation = store.game?.evaluation || {};
    const values = [
      { name: "推理一致性", value: metricValue(evaluation, "process_metrics.speech_vote_consistency", "speech_vote_consistency", "reasoning_consistency") },
      { name: "阵营协同度", value: metricValue(evaluation, "process_metrics.wolf_coordination", "wolf_coordination", "coordination") },
      { name: "投票命中率", value: metricValue(evaluation, "process_metrics.vote_accuracy", "vote_accuracy", "metrics.vote_accuracy") },
      { name: "信息利用率", value: metricValue(evaluation, "process_metrics.seer_check_utilization", "seer_check_utilization", "information_utilization") },
    ];
    const present = values.filter((item) => item.value !== null);
    const overall = present.length ? Math.round(present.reduce((sum, item) => sum + item.value, 0) / present.length) : null;
    $("#overall-score").textContent = overall === null ? "—" : String(overall);
    $(".metric-ring").style.setProperty("--score", overall || 0);
    $("#metric-bars").innerHTML = values.map((item) => `<div class="metric-row"><div><span>${item.name}</span><b>${item.value === null ? "—" : `${Math.round(item.value)}%`}</b></div><progress max="100" value="${item.value || 0}"></progress></div>`).join("");
    const model = firstDefined(store.game?.model, store.game?.config?.strategy, store.game?.players.find((item) => item.model)?.model, "Heuristic");
    $("#model-tag").textContent = String(model);
    const points = firstDefined(evaluation.turning_points, evaluation.replay?.key_turning_points, store.game?.turningPoints, []);
    const latest = Array.isArray(points) ? points.at(-1) : null;
    $("#turning-point").innerHTML = `<span>关键节点${latest?.seq ? ` · #${escapeHtml(latest.seq)}` : ""}</span><p>${escapeHtml(firstDefined(latest?.summary, latest?.description, latest?.reason, "对局开始后，将基于事件序列识别局势转折。"))}</p>`;
    renderSheriffMetrics(evaluation);
  }

  function renderSheriffMetrics(evaluation) {
    const card = $("#sheriff-metric-card");
    const sheriff = firstDefined(evaluation?.process_metrics?.sheriff, evaluation?.sheriff_metrics, evaluation?.sheriff);
    if (!sheriff || typeof sheriff !== "object") {
      card.hidden = true;
      card.innerHTML = "";
      return;
    }
    const election = sheriff.election || {};
    const leadership = sheriff.leadership || {};
    const flow = firstDefined(sheriff.badge_flow, sheriff.badgeFlow, {});
    const winner = election.winner_id ? playerName(election.winner_id) : election.void ? "竞选流局" : "尚未产生";
    const response = percent(leadership.nomination_response_rate);
    const flowRate = percent(flow.execution_rate);
    card.hidden = false;
    card.innerHTML = `<div class="sheriff-metric-title"><span>♛</span><strong>警长表现</strong><small>SCHEMA 1.1</small></div><div class="sheriff-metric-grid">
      <div><span>警徽归属</span><b>${escapeHtml(winner)}</b></div>
      <div><span>归票响应</span><b>${response === null ? "—" : `${Math.round(response)}%`}</b></div>
      <div><span>关键 1.5 票</span><b>${escapeHtml(firstDefined(leadership.decisive_weighted_vote_count, 0))}</b></div>
      <div><span>警徽流执行</span><b>${flowRate === null ? "—" : `${Math.round(flowRate)}%`}</b></div>
    </div>`;
  }

  function renderHumanTurn() {
    const panel = $("#human-turn");
    const pending = store.game?.pendingAction;
    const waiting = store.game?.status === "waiting_for_human" || Boolean(pending);
    const actor = firstDefined(pending?.actor_id, pending?.actor, store.game?.players.find((player) => player.controller === "human")?.id);
    const canAct = Boolean(actor) && String(store.perspective) === String(actor);
    panel.hidden = !waiting || !canAct;
    if (!waiting || !canAct) {
      closeHumanComposer({ restoreActions: false, restoreFocus: false });
      return;
    }
    $("#human-prompt").textContent = humanPrompt(pending, store.game.phase);
    $("#human-description").textContent = `当前阶段：${phaseInfo(firstDefined(pending?.phase, store.game.phase)).name}。行动提交后将由规则层校验。`;
    const actions = asArray(firstDefined(pending?.legal_actions, pending?.actions, []));
    const container = $("#human-actions");
    const pendingKey = [actor, firstDefined(pending?.phase, store.game.phase), firstDefined(pending?.round, store.game.round)].join("|");
    if (!actions.length) {
      container.innerHTML = `<button class="human-action" type="button" data-human-action="pass">${escapeHtml(passLabel(firstDefined(pending?.phase, store.game.phase)))}</button>`;
      container.__actions = [];
      closeHumanComposer({ restoreActions: true, restoreFocus: false });
    } else {
      const buttons = [];
      actions.forEach((action) => {
        const actionType = firstDefined(action.action_type, action.type, "pass");
        const targets = firstDefined(action.target_ids, action.targets, []);
        if (Array.isArray(targets) && targets.length) {
          targets.forEach((target) => buttons.push({ actionType, target, actor, pendingKey, contentRequired: action.content_required === true, description: action.description, label: `${actionLabel(actionType, pending?.phase)} · ${actionTargetPrefix(actionType, pending?.phase)}${playerName(target)}` }));
        } else {
          buttons.push({ actionType, target: null, actor, pendingKey, contentRequired: action.content_required === true, description: action.description, label: actionLabel(actionType, pending?.phase) });
        }
        if (action.can_pass) buttons.push({ actionType: "pass", target: null, actor, pendingKey, label: passLabel(firstDefined(pending?.phase, store.game.phase)) });
      });
      const uniqueButtons = buttons.filter((item, index, values) => values.findIndex((candidate) => humanActionKey(candidate) === humanActionKey(item)) === index);
      container.innerHTML = uniqueButtons.slice(0, 24).map((item, index) => `<button class="human-action${item.actionType === "sheriff_destroy" ? " danger" : isSheriffHumanAction(item.actionType, pending?.phase) ? " sheriff" : ""}" type="button" data-action-index="${index}">${escapeHtml(item.label)}</button>`).join("");
      container.__actions = uniqueButtons;
      const draftAction = store.humanDraft?.action;
      const currentDraftAction = draftAction && uniqueButtons.find((item) => humanActionKey(item) === humanActionKey(draftAction));
      if (currentDraftAction) {
        showHumanComposer(currentDraftAction, { focus: false, preserveContent: true });
      } else {
        closeHumanComposer({ restoreActions: true, restoreFocus: false });
      }
    }
  }

  function actionLabel(value, phase = "") {
    if (String(value) === "speak") {
      const phaseKey = String(phase || "");
      if (phaseKey === "sheriff_speech") return "警上发言";
      if (phaseKey === "sheriff_pk_speech") return "警长竞选 PK 发言";
      if (phaseKey === "day_pk_speech") return "放逐 PK 发言";
    }
    const labels = {
      wolf_kill: "袭击", seer_check: "查验", witch_save: "使用解药", witch_poison: "使用毒药",
      speak: "发言", vote: "投票", hunter_shoot: "开枪", pass: "弃权 / 不行动",
      sheriff_run: "举手上警", sheriff_withdraw: "退水", sheriff_vote: "投警长票",
      sheriff_order_clockwise: "顺时针发言", sheriff_order_counterclockwise: "逆时针发言",
      sheriff_summarize: "警长总结并归票", sheriff_transfer: "移交警徽", sheriff_destroy: "撕毁警徽",
    };
    return labels[String(value)] || String(value).replaceAll("_", " ");
  }

  function actionTargetPrefix(actionType, phase = "") {
    if (String(actionType) === "speak") return "重点提及 ";
    if (String(actionType) === "sheriff_summarize") return "归票 ";
    if (String(actionType) === "sheriff_transfer") return "传给 ";
    return "";
  }

  function isSheriffHumanAction(actionType, phase = "") {
    return String(actionType).startsWith("sheriff_") || ["sheriff_speech", "sheriff_pk_speech"].includes(String(phase || ""));
  }

  function humanPrompt(pending, fallbackPhase) {
    const phase = String(firstDefined(pending?.phase, fallbackPhase, ""));
    const prompts = {
      sheriff_run: "你是否举手竞选警长？",
      sheriff_speech: "请完成警上竞选发言",
      sheriff_withdraw: "你要继续竞选还是退水？",
      sheriff_vote: "请投出警长选票，或选择弃权",
      sheriff_pk_speech: "请完成警长竞选 PK 发言",
      sheriff_pk_vote: "请投出警长 PK 选票，或选择弃权",
      sheriff_badge: "请移交警徽或撕毁警徽",
      sheriff_order: "请选择今天的发言方向",
      day_discussion: "请完成本轮白天发言",
      day_vote: "请投出放逐票",
      day_pk_speech: "请完成放逐 PK 发言",
      day_pk_vote: "请投出放逐 PK 票",
      hunter_shot: "请选择猎枪目标，或放弃开枪",
    };
    return prompts[phase] || firstDefined(pending?.prompt, "轮到你行动");
  }

  function passLabel(phase) {
    const key = String(phase || "");
    if (key === "sheriff_run") return "不上警";
    if (key === "sheriff_withdraw") return "继续竞选";
    if (["sheriff_vote", "sheriff_pk_vote"].includes(key)) return "弃权（警长票）";
    return "弃权 / 不行动";
  }

  function humanActionKey(action) {
    return [action?.pendingKey, action?.actor, action?.actionType, action?.target ?? ""].join("|");
  }

  function showHumanComposer(action, { focus = true, preserveContent = false } = {}) {
    const previous = store.humanDraft;
    const sameAction = previous && humanActionKey(previous.action) === humanActionKey(action);
    const content = preserveContent && sameAction ? previous.content : "";
    store.humanDraft = { action, content };

    const isSpeech = ["speak", "sheriff_summarize"].includes(String(action.actionType));
    const composer = $("#human-composer");
    const textarea = $("#human-content");
    $("#human-actions").hidden = true;
    composer.hidden = false;
    const isSummary = String(action.actionType) === "sheriff_summarize";
    $("#human-content-label").textContent = isSummary ? "完成警长总结与归票发言" : isSpeech ? "组织你的公开发言" : `补充「${actionLabel(action.actionType, store.game?.pendingAction?.phase)}」内容`;
    $("#submit-human-content").textContent = isSummary ? "提交归票" : isSpeech ? "提交发言" : "提交行动";
    $("#human-content-hint").textContent = isSpeech
      ? "发言会进入公共频道。Ctrl / ⌘ + Enter 快速提交，Esc 取消。"
      : "内容会随本次行动一起提交。Ctrl / ⌘ + Enter 快速提交，Esc 取消。";
    textarea.placeholder = isSummary ? "总结场上信息，并说明你的放逐建议…" : isSpeech ? "陈述你的判断、质疑或身份信息…" : "输入本次行动所需的补充内容…";
    if (textarea.value !== content) textarea.value = content;
    updateHumanComposer();
    if (focus) requestAnimationFrame(() => textarea.focus());
  }

  function updateHumanComposer() {
    const textarea = $("#human-content");
    const content = textarea.value;
    if (store.humanDraft) store.humanDraft.content = content;
    $("#human-content-count").textContent = `${content.length} / ${textarea.maxLength}`;
    $("#submit-human-content").disabled = !content.trim();
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }

  function closeHumanComposer({ restoreActions = true, restoreFocus = false } = {}) {
    const draft = store.humanDraft;
    store.humanDraft = null;
    const composer = $("#human-composer");
    const textarea = $("#human-content");
    composer.hidden = true;
    textarea.value = "";
    textarea.style.height = "";
    $("#human-content-count").textContent = `0 / ${textarea.maxLength}`;
    $("#submit-human-content").disabled = true;
    const actions = $("#human-actions");
    actions.hidden = !restoreActions;
    if (restoreFocus && draft?.action) {
      const index = actions.__actions?.findIndex((item) => humanActionKey(item) === humanActionKey(draft.action));
      if (index >= 0) requestAnimationFrame(() => actions.querySelector(`[data-action-index="${index}"]`)?.focus());
    }
  }

  function renderControls() {
    const available = Boolean(store.game) && !isFinished(store.game);
    const autoRunning = store.auto || Boolean(store.game?.running);
    const waiting = store.game?.status === "waiting_for_human" || Boolean(store.game?.pendingAction);
    $("#step-game").disabled = !available || autoRunning || waiting;
    $("#run-game").disabled = !available || waiting;
    $("#run-icon").textContent = autoRunning ? "■" : "▶";
    $("#run-label").textContent = autoRunning ? "停止运行" : "自动运行";
    $("#open-current-replay").disabled = !store.game || !isFinished(store.game);
  }

  function renderMiniGame() {
    const card = $("#game-mini-card");
    if (!store.game) {
      card.innerHTML = `<div class="mini-empty"><span>尚未连接对局</span><small>创建或等待后端恢复</small></div>`;
      return;
    }
    const alive = store.game.players.filter((player) => player.alive).length;
    const progress = store.game.players.length ? Math.round((1 - alive / store.game.players.length) * 75 + Math.min(store.game.round * 5, 25)) : 0;
    card.innerHTML = `<div class="mini-game"><div class="mini-game-top"><strong>${escapeHtml(shortId(store.game.id))}</strong><span class="mini-phase">${escapeHtml(phaseInfo(store.game.phase).name)}</span></div><small>R${store.game.round || 0} · ${alive}/${store.game.players.length} 存活 · ${store.game.mode === "mixed" ? "人机混战" : "纯 AI"}</small><div class="mini-progress"><i style="width:${isFinished(store.game) ? 100 : clamp(progress)}%"></i></div></div>`;
  }

  function maybeShowResult() {
    if (!store.game || !isFinished(store.game) || store.resultShownFor === store.game.id) return;
    store.resultShownFor = store.game.id;
    archiveCurrentGame();
    const winner = winnerName(store.game.winner);
    $("#result-icon").textContent = store.game.winner?.includes("wolf") ? "♞" : "♜";
    $("#result-title").textContent = `${winner}获胜`;
    $("#result-description").textContent = firstDefined(store.game.evaluation?.result_metrics?.win_reason, store.game.winReason, `历经 ${store.game.round || 0} 个回合，胜负条件已经满足。`);
    const alive = store.game.players.filter((player) => player.alive).length;
    $("#result-stats").innerHTML = [
      [store.game.round || "—", "回合数"],
      [store.events.length, "结构化事件"],
      [`${alive}/${store.game.players.length}`, "最终存活"],
    ].map(([value, label]) => `<div class="result-stat"><strong>${escapeHtml(value)}</strong><span>${label}</span></div>`).join("");
    if (!$("#result-dialog").open) openModal($("#result-dialog"));
  }

  function archiveCurrentGame() {
    const game = store.game;
    if (!game || !isFinished(game)) return false;
    const item = {
      id: game.id,
      completed: true,
      winner: game.winner,
      round: game.round,
      players: game.players.length,
      playerStates: cloneProjection(game.players, []),
      sheriff: cloneProjection(game.sheriff, null),
      eventCount: store.events.length,
      score: Number($("#overall-score").textContent) || null,
      timestamp: new Date().toISOString(),
      perspective: store.perspective,
      evaluation: game.evaluation,
      events: store.events.slice(-200),
    };
    store.history = [item, ...store.history.filter((entry) => entry.id !== item.id)].slice(0, 20);
    saveHistory();
    renderReplayList();
    return true;
  }

  function renderReplayList() {
    const list = $("#replay-list");
    $("#history-count").textContent = String(store.history.length);
    if (!store.history.length) {
      list.innerHTML = `<div class="empty-state"><span class="empty-glyph">↺</span><strong>还没有可复盘的对局</strong><p>完成一场比赛后，胜负、关键事件和量化归因将保存在这里。</p></div>`;
      return;
    }
    list.innerHTML = store.history.map((item) => `<button class="replay-card" type="button" data-replay-id="${escapeHtml(item.id)}"><span class="replay-result-icon">${String(item.winner).includes("wolf") ? "♞" : "♜"}</span><span class="replay-copy"><strong>${escapeHtml(winnerName(item.winner))} · ${escapeHtml(shortId(item.id))}</strong><small>${escapeHtml(formatDate(item.timestamp))} · R${escapeHtml(item.round || "—")} · ${escapeHtml(item.eventCount || 0)} 事件</small></span><span class="replay-score">${item.score ?? "—"}</span></button>`).join("");
  }

  function showReplay(id) {
    const item = store.history.find((entry) => String(entry.id) === String(id));
    if (!item) return;
    $$(".replay-card").forEach((card) => card.classList.toggle("active", card.dataset.replayId === String(id)));
    const evaluation = item.evaluation || {};
    const points = firstDefined(evaluation.turning_points, evaluation.replay?.key_turning_points, []);
    const fallback = item.events.filter((event) => ["vote", "death", "game_over", "wolf_kill", "hunter_shoot", "day_pk"].some((key) => event.type.includes(key))).slice(-8).map((event) => ({ seq: event.seq, summary: event.message, impact: event.type }));
    const sheriffTypes = new Set(["sheriff_election_started", "sheriff_elected", "sheriff_election_void", "sheriff_order", "sheriff_summary", "badge_transferred", "badge_destroyed", "day_pk_started"]);
    const sheriffChain = item.events.filter((event) => sheriffTypes.has(event.type)).map((event) => ({ seq: event.seq, title: eventInfo(event.type).label, summary: event.message, category: event.type }));
    const baseRows = Array.isArray(points) && points.length ? points : fallback;
    const rows = [...baseRows, ...sheriffChain]
      .filter((item, index, values) => values.findIndex((candidate) => String(candidate.seq) === String(item.seq) && firstDefined(candidate.category, candidate.event_type, candidate.impact) === firstDefined(item.category, item.event_type, item.impact)) === index)
      .sort((a, b) => Number(a.seq || 0) - Number(b.seq || 0))
      .slice(-18);
    $("#attribution-content").className = "attribution-view";
    const sheriffSummary = replaySheriffSummary(evaluation, item);
    $("#attribution-content").innerHTML = `${sheriffSummary}${rows.length ? rows.map((point, index) => `<article class="trace-event ${String(firstDefined(point.category, point.event_type, "")).includes("sheriff") || String(firstDefined(point.category, point.event_type, "")).includes("badge") ? "sheriff" : ""}"><i>${index + 1}</i><strong>${escapeHtml(firstDefined(point.title, point.type, point.category, `关键事件 #${point.seq || index + 1}`))}</strong><p>${escapeHtml(firstDefined(point.summary, point.description, point.reason, "该事件改变了后续决策空间。"))}</p></article>`).join("") : `<div class="empty-state"><span class="empty-glyph">◇</span><strong>暂无可识别的转折点</strong><p>事件信息不足，仍可通过导出的结构化日志人工复核。</p></div>`}`;
  }

  function replaySheriffSummary(evaluation, item) {
    const sheriff = firstDefined(evaluation?.process_metrics?.sheriff, evaluation?.sheriff_metrics);
    if (!sheriff || typeof sheriff !== "object") return "";
    const election = sheriff.election || {};
    const leadership = sheriff.leadership || {};
    const flow = sheriff.badge_flow || {};
    const players = Array.isArray(item.playerStates) ? item.playerStates : [];
    const replayName = (playerId) => players.find((player) => String(player.id) === String(playerId))?.name || playerId || "—";
    const holder = election.winner_id ? replayName(election.winner_id) : election.void ? "竞选流局" : "未产生";
    const response = percent(leadership.nomination_response_rate);
    const flowRate = percent(flow.execution_rate);
    return `<section class="replay-sheriff-summary"><div><span>♛</span><strong>警长链路</strong></div><p>首任：${escapeHtml(holder)} · 归票响应：${response === null ? "—" : `${Math.round(response)}%`} · 关键 1.5 票：${escapeHtml(firstDefined(leadership.decisive_weighted_vote_count, 0))} · 警徽流：${flowRate === null ? "—" : `${Math.round(flowRate)}%`}</p></section>`;
  }

  function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.valueOf())) return "时间未知";
    return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false });
  }

  async function loadLeaderboard() {
    const button = $("#refresh-leaderboard");
    button.disabled = true;
    try {
      const data = await request("/api/leaderboard", { timeout: 6000 });
      store.leaderboard = data;
      renderLeaderboard();
    } catch (error) {
      renderLeaderboard();
      toast("榜单接口暂不可用", "完成多局竞技后，可由后端开放 /api/leaderboard。", "info");
    } finally {
      button.disabled = false;
    }
  }

  function renderLeaderboard(role = "all") {
    const data = store.leaderboard || {};
    let rows = firstDefined(data.rankings, data.entries, data.leaderboard, []);
    rows = asArray(rows);
    if (role !== "all") rows = rows.filter((item) => String(firstDefined(item.role, item.character, "")).toLowerCase() === role || item.roles?.[role]);
    $("#total-games").textContent = String(firstDefined(data.games_recorded, data.total_games, store.history.length, 0));
    $("#agent-count").textContent = String(rows.length);
    $("#top-elo").textContent = rows.length ? String(Math.round(Number(firstDefined(rows[0].rating, rows[0].elo, 1500)))) : "—";
    $("#top-agent").textContent = rows.length ? String(firstDefined(rows[0].name, rows[0].key, rows[0].agent, "Agent")) : "暂无数据";
    const tbody = $("#leaderboard-body");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6"><div class="table-empty">完成批量竞技后生成排行榜</div></td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map((item, index) => {
      const games = Number(firstDefined(item.games, item.game_count, 0));
      const winRate = percent(firstDefined(item.win_rate, item.wins && games ? item.wins / games : 0));
      const trend = Number(firstDefined(item.rating_change, item.trend, 0));
      return `<tr><td><span class="rank-medal">${escapeHtml(firstDefined(item.rank, index + 1))}</span></td><td><strong>${escapeHtml(firstDefined(item.name, item.key, item.agent, "Agent"))}</strong><br><small>${escapeHtml(firstDefined(item.version, item.model, "default"))}</small></td><td>${Math.round(Number(firstDefined(item.rating, item.elo, 1500)))}</td><td>${games}</td><td>${winRate === null ? "—" : `${Math.round(winRate)}%`}</td><td class="${trend >= 0 ? "trend-up" : "trend-down"}">${trend ? `${trend > 0 ? "+" : ""}${trend.toFixed(1)}` : "—"}</td></tr>`;
    }).join("");
  }

  async function exportLog() {
    if (!store.game) {
      toast("暂无日志", "请先创建或连接一场对局。", "info");
      return;
    }
    const gameId = store.game.id;
    const revision = store.viewRevision;
    const perspective = store.perspective;
    const stateRequestPath = statePath();
    const eventsRequestPath = eventsPath(gameId, 0);
    const button = $("#export-log");
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "准备当前视角…";
    try {
      const [statePayload, eventPayload] = await Promise.all([
        request(stateRequestPath, { timeout: 10000 }),
        request(eventsRequestPath, { timeout: 10000 }),
      ]);
      if (revision !== store.viewRevision || store.game?.id !== gameId) {
        toast("导出已取消", "准备日志期间视角发生变化，请重新导出。", "info");
        return;
      }
      const projectedGame = normalizeGame(statePayload);
      if (!projectedGame) throw new Error("后端没有返回可导出的当前视角状态");
      const projectedEvents = extractEvents(eventPayload).map(normalizeEvent);
      const payload = {
        schema_version: "1.1",
        exported_at: new Date().toISOString(),
        perspective,
        game: projectedGame,
        events: projectedEvents,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `werewolf-${gameId}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast("日志已导出", "已从后端重新获取并保存当前视角投影。", "success");
    } catch (error) {
      if (revision === store.viewRevision) toast("导出失败", humanizeError(error), "error");
    } finally {
      button.disabled = false;
      button.textContent = originalLabel;
    }
  }

  function toast(title, message, type = "info") {
    const region = $("#toast-region");
    const element = document.createElement("div");
    element.className = `toast ${type}`;
    const icon = type === "error" ? "!" : type === "success" ? "✓" : "i";
    element.innerHTML = `<span class="toast-icon">${icon}</span><div><strong>${escapeHtml(title)}</strong><small>${escapeHtml(message)}</small></div><button type="button" aria-label="关闭">×</button>`;
    $("button", element).addEventListener("click", () => element.remove());
    region.appendChild(element);
    window.setTimeout(() => element.remove(), type === "error" ? 7000 : 4200);
  }

  function playEventTone() {
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) return;
      const context = new AudioContextClass();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.frequency.value = 520;
      oscillator.type = "sine";
      gain.gain.setValueAtTime(0.035, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.12);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + 0.12);
      oscillator.addEventListener("ended", () => context.close());
    } catch {
      // Audio feedback is optional.
    }
  }

  function navigate(view) {
    $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
    $$(".page").forEach((page) => page.classList.toggle("active", page.dataset.page === view));
    const titles = { arena: "实时对局", replay: "复盘中心", leaderboard: "竞技榜单" };
    $("#page-title").textContent = titles[view] || "Agent Arena";
    document.body.classList.remove("sidebar-open");
    if (view === "leaderboard" && !store.leaderboard) loadLeaderboard();
    if (view === "replay") renderReplayList();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function bindEvents() {
    $("#open-create").addEventListener("click", () => openModal($("#create-dialog")));
    $("#create-form").addEventListener("submit", (event) => {
      const submitter = event.submitter;
      event.preventDefault();
      if (submitter?.value === "cancel") {
        closeModal($("#create-dialog"));
        return;
      }
      createGame(event.currentTarget);
    });
    $$("input[name='mode']").forEach((input) => input.addEventListener("change", () => {
      $$(".mode-card").forEach((card) => card.classList.toggle("selected", $("input", card).checked));
      $("#human-config").hidden = input.value !== "mixed" || !input.checked;
    }));
    $("#strategy-select").addEventListener("change", (event) => {
      event.currentTarget.dataset.userSelected = "true";
    });
    $$(".nav-item").forEach((item) => item.addEventListener("click", () => navigate(item.dataset.view)));
    $$(".chip").forEach((chip) => chip.addEventListener("click", () => {
      store.overlay = chip.dataset.overlay;
      $$(".chip").forEach((item) => item.classList.toggle("active", item === chip));
      renderSeats();
    }));
    $$("[data-event-filter]").forEach((button) => button.addEventListener("click", () => {
      store.eventFilter = button.dataset.eventFilter;
      $$("[data-event-filter]").forEach((item) => item.classList.toggle("active", item === button));
      renderEvents();
    }));
    $("#step-game").addEventListener("click", () => stepGame());
    $("#run-game").addEventListener("click", runGame);
    $("#retry-connection").addEventListener("click", () => loadState());
    $("#jump-latest").addEventListener("click", () => $("#event-stream").scrollTo({ top: $("#event-stream").scrollHeight, behavior: "smooth" }));
    $("#export-log").addEventListener("click", exportLog);
    $("#sound-toggle").addEventListener("click", (event) => {
      store.sound = !store.sound;
      event.currentTarget.textContent = store.sound ? "♫" : "♬";
      event.currentTarget.style.color = store.sound ? "var(--gold-bright)" : "";
      toast(store.sound ? "事件提示音已开启" : "事件提示音已关闭", store.sound ? "新事件到达时会播放轻提示音。" : "观战将保持静音。", "info");
    });
    $("#theme-toggle").addEventListener("click", (event) => {
      store.effects = !store.effects;
      document.body.classList.toggle("reduced-effects", !store.effects);
      event.currentTarget.textContent = store.effects ? "◐" : "○";
    });
    $("#perspective-select").addEventListener("change", async (event) => {
      store.perspective = event.target.value;
      store.viewRevision += 1;
      store.runController?.abort();
      store.runController = null;
      // Logs are server-projected per viewer. Never carry omniscient/private
      // events across a perspective change, even briefly in the DOM.
      store.events = [];
      store.eventKeys.clear();
      store.game = transitionGameSnapshot(store.game);
      store.auto = Boolean(store.game?.running);
      clearInterval(store.pollTimer);
      closeSocket();
      renderAll();
      try {
        await loadState({ quiet: true });
        await pollEvents({ quiet: true });
        toast("视角已切换", event.target.options[event.target.selectedIndex].textContent, "success");
      } catch (error) {
        toast("视角切换失败", humanizeError(error), "error");
      }
    });
    $("#human-actions").addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      const pending = store.game?.pendingAction || {};
      if (button.dataset.humanAction === "pass") {
        $$('button', event.currentTarget).forEach((item) => { item.disabled = true; });
        stepGame({ actor_id: pending.actor_id, action_type: "pass" });
        return;
      }
      const item = event.currentTarget.__actions?.[Number(button.dataset.actionIndex)];
      if (!item) return;
      if (item.contentRequired) {
        showHumanComposer(item);
        return;
      }
      $$('button', event.currentTarget).forEach((actionButton) => { actionButton.disabled = true; });
      stepGame({ actor_id: item.actor, action_type: item.actionType, target_id: item.target, content: null });
    });
    $("#human-content").addEventListener("input", updateHumanComposer);
    $("#human-content").addEventListener("keydown", (event) => {
      if (event.isComposing) return;
      if (event.key === "Escape") {
        event.preventDefault();
        closeHumanComposer({ restoreActions: true, restoreFocus: true });
      } else if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        if (!$("#submit-human-content").disabled) $("#human-composer").requestSubmit();
      }
    });
    $("#cancel-human-content").addEventListener("click", () => closeHumanComposer({ restoreActions: true, restoreFocus: true }));
    $("#human-composer").addEventListener("submit", (event) => {
      event.preventDefault();
      const draft = store.humanDraft;
      const content = $("#human-content").value.trim();
      if (!draft?.action || !content) return;
      const action = draft.action;
      closeHumanComposer({ restoreActions: true, restoreFocus: false });
      $$("button", $("#human-actions")).forEach((button) => { button.disabled = true; });
      stepGame({ actor_id: action.actor, action_type: action.actionType, target_id: action.target, content });
    });
    $("#mobile-menu").addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
    $("#mobile-scrim").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
    $("#close-result").addEventListener("click", () => closeModal($("#result-dialog")));
    $("#go-replay").addEventListener("click", () => { closeModal($("#result-dialog")); navigate("replay"); if (store.game) showReplay(store.game.id); });
    $("#open-current-replay").addEventListener("click", () => {
      if (!store.game || !isFinished(store.game)) {
        toast("对局尚未结束", "完成胜负裁定后才能生成持久化复盘。", "info");
        return;
      }
      if (!store.history.some((item) => item.id === store.game.id)) archiveCurrentGame();
      showReplay(store.game.id);
    });
    $("#replay-list").addEventListener("click", (event) => { const card = event.target.closest("[data-replay-id]"); if (card) showReplay(card.dataset.replayId); });
    $("#refresh-leaderboard").addEventListener("click", loadLeaderboard);
    $("#role-tabs").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-role]");
      if (!button) return;
      $$("button", event.currentTarget).forEach((item) => item.classList.toggle("active", item === button));
      renderLeaderboard(button.dataset.role);
    });
    document.addEventListener("visibilitychange", () => { if (!document.hidden && store.game) pollEvents({ quiet: true }); });
    window.addEventListener("beforeunload", () => { closeSocket(); clearInterval(store.pollTimer); store.runController?.abort(); });
  }

  async function initialize() {
    bindEvents();
    renderAll();
    try {
      await loadCapabilities();
    } catch {
      applyProviderCapabilities();
    }
    try {
      await loadState();
    } catch {
      // The visible offline state already explains how to proceed.
    }
  }

  initialize();
})();
