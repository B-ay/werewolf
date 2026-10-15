/*
 * 浏览器回归：重复快照、按区域更新、节点复用、历史阅读位置、视角隔离和 HTTP 同步。
 * 运行：在可解析 playwright 的 Node 环境中执行 node tests/frontend_sync.spec.cjs。
 * 使用独立的无头浏览器和本地请求替身，不调用真实 DeepSeek，也不修改用户对局。
 */
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { chromium } = require("playwright");

async function waitUntil(predicate, label) {
  const deadline = Date.now() + 7000;
  while (!predicate()) {
    assert.ok(Date.now() < deadline, `Timeout: ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
}

async function main() {
  const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || "msedge", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    const requests = { state: 0, events: 0, sockets: 0, active: 0, maxActive: 0 };
    let nextStateGate = null;
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("websocket", () => { requests.sockets += 1; });
    let state = {
      game_id: "sync-regression", round: 1, stage: "day_discussion", phase: "day",
      status: "running", running: false, model: "heuristic", winner: null,
      config: { enable_sheriff: false }, sheriff: { enabled: false },
      players: Array.from({ length: 8 }, (_, index) => ({
        id: `p${index + 1}`, name: `Agent ${index + 1}`, seat: index + 1,
        alive: true, role: index === 0 ? "seer" : "villager", controller: "ai",
      })),
      events: Array.from({ length: 75 }, (_, index) => ({
        seq: index + 1, round: 1, phase: "day_discussion", event_type: "speech",
        actor_id: `p${index % 8 + 1}`, message: `历史发言 ${index + 1}：${"这是可见的公开信息。".repeat(8)}`,
        data: {}, visibility: ["public"],
      })),
    };
    await page.route("**/api/**", async (route) => {
      const url = new URL(route.request().url());
      let data;
      if (url.pathname === "/api/health") {
        data = { ok: true, transport: { type: "http_polling", poll_interval_ms: 1000 }, providers: {} };
      } else if (url.pathname === "/api/state") {
        requests.state += 1;
        requests.active += 1;
        requests.maxActive = Math.max(requests.maxActive, requests.active);
        // 验证仅 updated_at 改变时不触发 DOM 更新。
        data = { ...structuredClone(state), updated_at: new Date().toISOString() };
        const viewer = url.searchParams.get("viewer");
        if (url.searchParams.get("omniscient") !== "true") {
          data.players = data.players.map((player) => ({ ...player, role: player.id === viewer ? player.role : null }));
          data.events = data.events.filter((event) => event.visibility.includes("public") || event.visibility.includes(viewer));
        }
        if (nextStateGate) {
          const gate = nextStateGate;
          nextStateGate = null;
          await gate;
        }
      } else if (url.pathname.endsWith("/step")) {
        data = structuredClone(state);
      } else {
        requests.events += 1;
        data = { events: [] };
      }
      await route.fulfill({ status: 200, contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" }, body: JSON.stringify(data) });
      if (url.pathname === "/api/state") requests.active -= 1;
    });
    await page.goto(pathToFileURL(path.resolve(__dirname, "../frontend/index.html")).href);
    await page.waitForFunction(() => document.querySelector("#event-count").textContent === "75");
    await page.addStyleTag({ content: "* { animation: none !important; scroll-behavior: auto !important; }" });
    await page.evaluate(() => {
      const stream = document.querySelector("#event-stream");
      const conversations = document.querySelector("#conversation-list");
      stream.scrollTop = 135;
      conversations.scrollTop = 210;
      window.syncRefs = {
        seat: document.querySelector(".player-seat"), avatar: document.querySelector(".seat-avatar"),
        event: stream.children[0], speech: conversations.children[0],
        eventTop: stream.scrollTop, speechTop: conversations.scrollTop,
        mutations: 0,
      };
      window.syncObserver = new MutationObserver((records) => { window.syncRefs.mutations += records.length; });
      window.syncObserver.observe(document.body, { childList: true, subtree: true, attributes: true, characterData: true });
    });
    let before = requests.state;
    await waitUntil(() => requests.state >= before + 2, "two unchanged polls");
    await page.evaluate(() => {
      const refs = window.syncRefs;
      if (refs.mutations !== 0) throw new Error(`Unchanged snapshot mutated DOM ${refs.mutations} times`);
      if (document.querySelector("#event-stream").scrollTop !== refs.eventTop) throw new Error("Reading position changed");
    });
    console.log("PASS: repeated snapshots and timestamp-only changes leave DOM untouched");

    state.events.push({ seq: 76, round: 1, event_type: "vote", actor_id: "p1", target_id: "p2", message: "一条新投票", data: {}, visibility: ["public"] });
    await page.waitForFunction(() => document.querySelector("#event-count").textContent === "76");
    await page.evaluate(() => {
      const refs = window.syncRefs;
      if (document.querySelector(".player-seat") !== refs.seat || document.querySelector(".seat-avatar") !== refs.avatar) throw new Error("Seat nodes replaced");
      if (document.querySelector("#event-stream").children[0] !== refs.event) throw new Error("Old event replaced");
      if (document.querySelector("#conversation-list").children[0] !== refs.speech) throw new Error("Unrelated speech replaced");
      if (Math.abs(document.querySelector("#event-stream").scrollTop - refs.eventTop) > 1) throw new Error("Event reader dragged to bottom");
      if (document.querySelector("#conversation-list").scrollTop !== refs.speechTop) throw new Error("Speech reading position changed");
      window.syncObserver.disconnect();
    });
    console.log("PASS: new event preserves seats, existing rows, and reading positions");

    await page.evaluate(() => {
      const stream = document.querySelector("#event-stream");
      const conversations = document.querySelector("#conversation-list");
      stream.scrollTop = stream.scrollHeight;
      conversations.scrollTop = conversations.scrollHeight;
    });
    state.events.push({ seq: 77, round: 1, event_type: "speech", actor_id: "p2", message: "新增的公开发言。".repeat(10), data: {}, visibility: ["public"] });
    await page.waitForFunction(() => document.querySelector("#event-count").textContent === "77");
    await page.evaluate(() => {
      for (const selector of ["#event-stream", "#conversation-list"]) {
        const element = document.querySelector(selector);
        if (element.scrollHeight - element.clientHeight - element.scrollTop > 2) throw new Error(`Not following bottom: ${selector}`);
      }
    });
    console.log("PASS: readers already at the bottom follow new events and speeches");

    // 列表窗口删除头部历史后，保留仍在窗口内的可见条目及其屏幕位置。
    await page.evaluate(() => {
      const stream = document.querySelector("#event-stream");
      stream.scrollTop += stream.children[50].getBoundingClientRect().top - stream.getBoundingClientRect().top;
      const top = stream.getBoundingClientRect().top;
      const anchor = [...stream.children].find((node) => node.getBoundingClientRect().bottom > top);
      window.windowAnchor = { node: anchor, top: anchor.getBoundingClientRect().top };
    });
    for (let seq = 78; seq <= 160; seq += 1) state.events.push({ seq, round: 1, event_type: "vote", actor_id: "p1", message: `新增事件 ${seq}`, data: {}, visibility: ["public"] });
    await page.waitForFunction(() => document.querySelector("#event-count").textContent === "160");
    await page.evaluate(() => {
      const { node, top } = window.windowAnchor;
      if (!node.isConnected || Math.abs(node.getBoundingClientRect().top - top) > 2) throw new Error(`Window trimming lost reading anchor: connected=${node.isConnected}, drift=${node.getBoundingClientRect().top - top}`);
    });
    console.log("PASS: event window trimming retains the visible reading anchor");

    // 指标变动不应触碰事件和座位。
    const refs = await page.evaluateHandle(() => ({ seat: document.querySelector(".player-seat"), event: document.querySelector("#event-stream").children[0] }));
    state.evaluation = { process_metrics: { speech_vote_consistency: { rate: 0.75 } } };
    await page.waitForFunction(() => document.querySelector("#overall-score").textContent === "75");
    await page.evaluate((previous) => {
      if (previous.seat !== document.querySelector(".player-seat") || previous.event !== document.querySelector("#event-stream").children[0]) throw new Error("Metrics updated unrelated nodes");
    }, refs);
    console.log("PASS: metric changes update their region only");

    // 请求悬挂期间的额外同步应合并；较早的轮询响应不能覆盖已返回的单步结果。
    let releaseState;
    nextStateGate = new Promise((resolve) => { releaseState = resolve; });
    await waitUntil(() => requests.active === 1, "held snapshot request");
    before = requests.state;
    await page.evaluate(() => document.dispatchEvent(new Event("visibilitychange")));
    await page.waitForTimeout(1200);
    assert.equal(requests.state, before, "Slow request should coalesce additional syncs");
    state.round = 2;
    await page.locator("#step-game").click();
    await page.waitForFunction(() => document.querySelector("#round-number").textContent === "02");
    releaseState();
    await waitUntil(() => requests.active === 0, "held snapshot completion");
    await page.waitForTimeout(100);
    assert.equal(await page.locator("#round-number").textContent(), "02", "Old poll overwrote newer action result");
    console.log("PASS: slow requests coalesce and stale polls cannot overwrite action results");

    // 人类发言草稿、焦点和选区在无关事件到来时必须保持。
    state.players[0].controller = "human";
    state.status = "waiting_for_human";
    state.pending_action = { actor_id: "p1", phase: "day_discussion", round: 2, legal_actions: [{ action_type: "speak", content_required: true, target_ids: [] }] };
    state.events.push({ seq: 161, round: 1, event_type: "seer_check", actor_id: "p1", message: "私有查验测试", data: {}, visibility: ["p1"] });
    await page.selectOption("#perspective-select", "p1");
    await page.locator("#human-actions button").first().click();
    await page.locator("#human-content").fill("我正在编辑尚未提交的发言");
    await page.evaluate(() => {
      const input = document.querySelector("#human-content");
      input.setSelectionRange(2, 5);
      window.draftRefs = { input, button: document.querySelector("#human-actions button") };
    });
    state.events.push({ seq: 162, round: 1, event_type: "vote", actor_id: "p2", message: "草稿期间新增事件", data: {}, visibility: ["public"] });
    await page.waitForFunction(() => document.querySelector("#event-count").textContent === "162");
    await page.evaluate(() => {
      const input = document.querySelector("#human-content");
      if (input !== window.draftRefs.input || document.activeElement !== input || input.selectionStart !== 2 || input.selectionEnd !== 5 || input.value !== "我正在编辑尚未提交的发言") throw new Error("Draft or focus lost");
      if (window.draftRefs.button !== document.querySelector("#human-actions button")) throw new Error("Unchanged human button replaced");
    });
    await page.selectOption("#perspective-select", "public");
    await page.waitForFunction(() => !document.querySelector("#event-stream").textContent.includes("私有查验测试") && document.querySelector("#human-turn").hidden);
    assert.equal(await page.locator(".player-seat.seer").count(), 0);
    console.log("PASS: drafts and focus survive unrelated changes; perspective switch clears secrets");

    state.pending_action = null;
    state.status = "finished";
    state.stage = "finished";
    state.winner = "village";
    await page.waitForFunction(() => document.querySelector("#phase-label").textContent === "对局结束");
    before = requests.state;
    await page.waitForTimeout(2300);
    assert.equal(requests.state, before, "Finished game should stop polling");
    assert.equal(requests.events, 0, "Snapshot sync should not redundantly fetch events");
    assert.equal(requests.sockets, 0, "HTTP-only server should never receive WebSocket attempts");
    assert.equal(requests.maxActive, 1, "Snapshot requests should not overlap");
    assert.deepEqual(errors, []);
    console.log("PASS: HTTP-only non-overlapping sync stops after the final snapshot; no browser errors");

    // 同样的节点复用行为也必须适用于移动布局。
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.locator(".player-seat").count(), 8);
    assert.equal(await page.locator("#connection-detail").textContent(), "HTTP 状态同步");
    console.log("PASS: mobile layout retains all seats and the stable connection indicator");
  } finally {
    await browser.close();
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
