"""启动 V3 API 服务

必须通过 bootstrap.py 启动，以正确初始化 DI 容器和模块路径
"""
import uvicorn


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 正在启动 V3 API 服务...")
    print("=" * 80)
    
    # 注意：此文件必须通过 bootstrap.py 启动
    # 使用方法: python src/bootstrap.py start_server.py
    
    # 启动 uvicorn 服务器
    uvicorn.run(
        "app:app",       # FastAPI 应用实例（已通过 bootstrap 初始化）
        host="0.0.0.0",  # 监听所有网络接口
        port=8001,       # 服务端口
        reload=False,    # 生产环境禁用自动重载
        log_level="info",
        access_log=True,
    )
