"""本项目评测与榜单的最小公共入口。"""
from .evaluator import evaluate_game
from .leaderboard import EloLeaderboard, build_leaderboard

__all__ = ["evaluate_game", "EloLeaderboard", "build_leaderboard"]
