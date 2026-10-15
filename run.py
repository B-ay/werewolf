"""狼人杀 Agent Team 的最小启动入口。

实际的参数解析、DeepSeek 环境变量读取、HTTP 服务创建和静态前端托管均在
``backend.server.main`` 中完成；本文件保持极薄，便于直接运行 ``python run.py``。
"""

from backend.server import main


if __name__ == "__main__":
    main()
