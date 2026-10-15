/*
 * 真实后端浏览器烟测：桌面/移动端纯 AI、人机操作、视角隔离与完局复盘。
 * Node 环境需能解析 playwright；不调用 DeepSeek，不接触用户正在运行的对局。
 * 独立 Python 服务使用随机端口和临时回放目录，退出后自动清理。
 */
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const readline = require("node:readline");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const serverScript = [
  "import json, sys, tempfile, threading",
  "from pathlib import Path",
  "from backend import server",
  "with tempfile.TemporaryDirectory(prefix='werewolf-smoke-') as directory:",
  "    server.DATA_ROOT = Path(directory)",
  "    manager = server.ArenaManager(deepseek_provider=None)",
  "    http = server.create_server('127.0.0.1', 0, manager=manager)",
  "    print(json.dumps({'port': http.server_port}), flush=True)",
  "    def stop():",
  "        sys.stdin.readline()",
  "        http.shutdown()",
  "    threading.Thread(target=stop, daemon=True).start()",
  "    try:",
  "        http.serve_forever()",
  "    finally:",
  "        if manager._evolution is not None: manager._evolution.wait(45)",
  "        http.server_close()",
].join("\n");

async function main() {
  const child = spawn(process.env.PYTHON || "python", ["-u", "-c", serverScript], {
    cwd: root, windowsHide: true,
    env: { ...process.env, DEEPSEEK_API_KEY: "" },
  });
  let diagnostics = "";
  child.stderr.on("data", (chunk) => { diagnostics += chunk.toString(); });
  let browser;
  try {
    const port = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Smoke server startup timed out")), 10000);
      const lines = readline.createInterface({ input: child.stdout });
      lines.once("line", (line) => { clearTimeout(timeout); resolve(JSON.parse(line).port); });
      child.once("error", (error) => { clearTimeout(timeout); reject(error); });
      child.once("exit", (code) => { clearTimeout(timeout); reject(new Error("Smoke server exited: " + code)); });
    });
    const base = "http://127.0.0.1:" + port;
    browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || "msedge", headless: true });
    for (const viewport of [{ width: 1440, height: 1100 }, { width: 390, height: 844 }]) {
      for (const mode of ["ai", "mixed"]) {
        const page = await browser.newPage({ viewport });
        const errors = [];
        page.on("pageerror", (error) => errors.push(error.message));
        const startup = page.waitForResponse((response) => response.url().startsWith(base + "/api/state"));
        await page.goto(base);
        await startup;
        await page.waitForFunction(() => document.querySelector("#connection-detail").textContent === "HTTP 状态同步");
        // 新页面会恢复服务端最近一局；先关闭该局的结果弹窗，再创建独立场景。
        await page.waitForTimeout(100);
        if (await page.locator("#result-dialog").isVisible()) await page.locator("#close-result").click();
        if (viewport.width < 600) await page.locator("#mobile-menu").click();
        await page.locator("#open-create").click();
        await page.locator('input[name="mode"][value="' + mode + '"]').check();
        await page.locator('[name="player_count"]').selectOption("6");
        await page.locator('[name="seed"]').fill("42");
        await page.locator("#strategy-select").selectOption("heuristic");
        if (mode === "ai") {
          await page.locator('label.rule-toggle:has([name="enable_evolution"])').click();
          assert.equal(await page.locator('[name="enable_evolution"]').isChecked(), true);
        }
        if (mode === "mixed") await page.locator('[name="human_seat"]').selectOption("1");
        const created = page.waitForResponse((response) => response.url() === base + "/api/games" && response.request().method() === "POST");
        await page.locator("#create-submit").click();
        const initial = await (await created).json();
        const id = initial.game_id;
        const viewer = mode === "mixed" ? "p1" : "observer";
        const query = "?viewer=" + viewer + "&omniscient=" + (mode === "ai");
        let state = initial;
        assert.equal(initial.config.enable_sheriff, true);
        assert.equal(Boolean(initial.evolution?.enabled), mode === "ai");
        await page.waitForFunction(() => document.querySelectorAll(".player-seat").length === 6);

        // 首次人类回合由实际 UI 提交；其余回合快速驱动真实引擎完成，以覆盖存档/复盘。
        let submittedFromUI = false;
        for (let step = 0; step < 200 && state.status !== "finished"; step += 1) {
          const pending = state.pending_action;
          if (pending) {
            assert.equal(pending.actor_id, "p1");
            // 当前默认规则公开死者身份；这不属于存活玩家的隐藏身份泄漏。
            assert.ok(state.players.every((player) => player.id === "p1" || !player.role || (!player.alive && state.config.reveal_roles_on_death)));
            if (!submittedFromUI) {
              await page.waitForFunction(() => !document.querySelector("#human-turn").hidden);
              const submitted = page.waitForResponse((response) => /\/(?:step|(?:human-)?action)$/.test(response.url()) && response.request().method() === "POST");
              await page.locator("#human-actions button").first().click();
              if (await page.locator("#human-composer").isVisible()) {
                await page.locator("#human-content").fill("我将依据当前可见的信息继续判断。");
                await page.locator("#submit-human-content").click();
              }
              state = await (await submitted).json();
              submittedFromUI = true;
              continue;
            }
            const rule = pending.legal_actions[0];
            const action = rule.can_pass
              ? { actor_id: "p1", action_type: "pass" }
              : { actor_id: "p1", action_type: rule.action_type, target_id: rule.target_ids?.[0] ?? null, content: rule.content_required ? "我会依据公开发言作出本轮判断。" : null };
            const response = await page.request.post(base + "/api/games/" + id + "/action", { data: { action, viewer: "p1", omniscient: false } });
            assert.ok(response.ok());
            state = await response.json();
          } else {
            const response = await page.request.post(base + "/api/games/" + id + "/step", { data: { viewer, omniscient: mode === "ai" } });
            assert.ok(response.ok());
            state = await response.json();
          }
        }
        assert.equal(state.status, "finished");
        if (mode === "mixed") assert.equal(submittedFromUI, true);
        await page.waitForFunction(() => document.querySelector("#result-dialog").open);
        await page.locator("#go-replay").click();
        await page.waitForFunction(() => document.querySelector("#attribution-content").classList.contains("attribution-view"));
        const reportResponse = await page.request.get(base + "/api/games/" + id + "/evaluation" + query);
        const report = await reportResponse.json();
        assert.equal(report.schema_version, "1.1");
        assert.equal(report.result_metrics.winner, state.winner);
        if (mode === "ai") {
          assert.equal(report.evolution.recorded, true);
          assert.equal(report.evolution.validation_status, "running");
          const evolution = await (await page.request.get(base + "/api/evolution")).json();
          assert.ok(evolution.scopes[initial.evolution.scope].training_game_count >= 1);
          assert.ok((await page.locator("#result-description").textContent()).includes("经验已记录"));
        }
        // 榜单也由真实评测数据生成，桌面与移动端都通过 UI 导航验证。
        if (viewport.width < 600) await page.locator("#mobile-menu").click();
        await page.locator('.nav-item[data-view="leaderboard"]').click();
        const rankings = page.waitForResponse((response) => response.url() === base + "/api/leaderboard");
        await page.locator("#refresh-leaderboard").click();
        const board = await (await rankings).json();
        assert.ok(board.rankings.length > 0);
        await page.waitForFunction(() => document.querySelector("#leaderboard-body .rank-medal") !== null);
        assert.deepEqual(errors, []);
        console.log("PASS: real backend " + mode + " at " + viewport.width + "px; sheriff, human actions, privacy, replay and leaderboard");
        await page.close();
      }
    }
  } catch (error) {
    console.error(diagnostics.slice(-4000));
    throw error;
  } finally {
    if (browser) await browser.close();
    if (child.exitCode === null) {
      const closed = new Promise((resolve) => child.once("exit", resolve));
      child.stdin.end("\n");
      const timeout = setTimeout(() => child.kill(), 60000);
      await closed;
      clearTimeout(timeout);
    }
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
