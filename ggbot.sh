#!/bin/bash
# GGBot 智能客服系统 — 快捷启动脚本
# 用法: ./ggbot.sh [命令]
# 无参数运行时显示交互菜单

set -e

# ── 配置 ──────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
title() { echo -e "\n${BOLD}${CYAN}═══ $1 ═══${NC}\n"; }

# ── 前置检查 ──────────────────────────────────────────────────────────────────
check_docker() {
    if ! command -v docker &>/dev/null; then
        error "Docker 未安装或未启动。请先启动 Docker Desktop。"
        echo -e "  提示：如果是 WSL，检查 Docker Desktop → Settings → WSL Integration 是否开启"
        exit 1
    fi
    if ! docker info &>/dev/null 2>&1; then
        error "Docker daemon 未运行。请先启动 Docker Desktop。"
        exit 1
    fi
}

check_env() {
    if [ ! -f .env ]; then
        warn ".env 文件不存在，从 .env.example 复制..."
        cp .env.example .env
        warn "请编辑 .env 填入真实 API key 后重新运行：ANTHROPIC_API_KEY / SILICONFLOW_API_KEY"
        exit 1
    fi
    # 检查是否还是占位符
    if grep -q "your_anthropic_api_key_here\|your_siliconflow_api_key_here" .env 2>/dev/null; then
        warn "检测到 .env 中仍有占位符 API key，请先编辑 .env 填入真实值："
        echo -e "  - ANTHROPIC_API_KEY  （DeepSeek 的 sk-xxx）"
        echo -e "  - SILICONFLOW_API_KEY （SiliconFlow 的 sk-xxx）"
        echo -e "  - ANTHROPIC_BASE_URL  （取消注释，指向 DeepSeek）"
        exit 1
    fi
}

# ── 核心命令 ──────────────────────────────────────────────────────────────────

cmd_start() {
    title "启动 GGBot"
    check_docker; check_env
    info "构建镜像（如有更新）..."
    docker compose build ggbot 2>&1 | tail -3
    info "启动所有服务..."
    docker compose up -d
    echo ""
    cmd_status
    echo ""
    info "✅ 启动完成！"
    echo -e "  API:      ${BLUE}http://localhost:8000${NC}"
    echo -e "  Swagger:  ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  CLI 对话: ${BLUE}./ggbot.sh cli${NC}"
}

cmd_stop() {
    title "停止 GGBot"
    check_docker
    docker compose stop
    info "已停止所有服务（数据保留）"
}

cmd_restart() {
    title "重启 GGBot"
    check_docker; check_env
    docker compose restart
    info "已重启"
    cmd_status
}

cmd_status() {
    title "服务状态"
    docker compose ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || docker compose ps
}

cmd_logs() {
    local svc="${1:-ggbot}"
    title "日志: $svc（Ctrl+C 退出）"
    docker compose logs -f "$svc"
}

cmd_cli() {
    title "CLI 交互模式"
    check_docker
    if ! docker ps --format '{{.Names}}' | grep -q "^ggbot-app$"; then
        error "ggbot-app 容器未运行，请先执行: ./ggbot.sh start"
        exit 1
    fi
    info "进入交互对话（输入 '退出' 或 Ctrl+C 结束）"
    echo ""
    docker exec -it ggbot-app python -m api.main --cli
}

cmd_health() {
    title "健康检查"
    local ok=true
    for ep in "health:8000"; do
        local name="${ep%%:*}"; local port="${ep##*:}"
        local code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/$name" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo -e "  ${GREEN}✓${NC} /$name  (HTTP $code)"
        else
            echo -e "  ${RED}✗${NC} /$name  (HTTP $code)  ← http://localhost:$port/$name"
            ok=false
        fi
    done
    echo ""
    if $ok; then info "所有检查通过"; else warn "部分检查未通过"; fi
}

cmd_test() {
    title "对话测试"
    local msg="${1:-你好，请一句话介绍退款政策}"
    info "发送: $msg"
    echo ""
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d "{\"message\":\"$msg\"}" \
        | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f'GGBot [{d.get(\"agent_type\",\"?\")}]: {d.get(\"response\",\"\")[:300]}')
    print(f'---')
    print(f'延迟: {d.get(\"latency_ms\")}ms  知识库: {d.get(\"knowledge_used\")}  状态: {d.get(\"status\")}')
except Exception as e:
    print(f'解析失败: {e}')
"
}

cmd_build() {
    title "重新构建镜像"
    check_docker
    docker compose build ggbot
    info "构建完成，重启容器..."
    docker compose up -d ggbot
}

cmd_down() {
    title "停止并删除容器（保留数据）"
    check_docker
    read -rp "确认停止并删除所有容器？(y/N) " confirm
    [ "$confirm" = "y" ] || { echo "已取消"; exit 0; }
    docker compose down
    info "已删除容器（数据卷保留）"
}

cmd_clean() {
    title "⚠️  彻底清理（删除容器+数据）"
    check_docker
    echo -e "${RED}此操作将删除所有容器、网络和数据卷，不可恢复！${NC}"
    read -rp "确认彻底清理？输入 'DELETE' 确认: " confirm
    [ "$confirm" = "DELETE" ] || { echo "已取消"; exit 0; }
    docker compose down -v
    info "已彻底清理"
}

cmd_shell() {
    title "进入 ggbot-app 容器 Shell"
    check_docker
    docker exec -it ggbot-app bash
}

# ── 帮助 ──────────────────────────────────────────────────────────────────────
show_help() {
    cat << 'EOF'

  ╔═══════════════════════════════════════════════════════════════════╗
  ║          GGBot 智能客服系统 — 快捷启动脚本                        ║
  ╚═══════════════════════════════════════════════════════════════════╝

  常用命令:
    ./ggbot.sh start     启动所有服务（首次会自动构建镜像）
    ./ggbot.sh cli       进入 CLI 交互对话模式
    ./ggbot.sh status    查看容器运行状态
    ./ggbot.sh logs      查看应用日志（实时滚动，Ctrl+C 退出）
    ./ggbot.sh test      发送测试对话（验证 DeepSeek + RAG）
    ./ggbot.sh health    执行健康检查

  管理命令:
    ./ggbot.sh stop      停止所有服务（数据保留）
    ./ggbot.sh restart   重启所有服务
    ./ggbot.sh build     重新构建应用镜像（改了代码后用）
    ./ggbot.sh down      停止并删除容器（数据卷保留）
    ./ggbot.sh clean     ⚠️ 彻底清理（删除容器+数据，不可恢复）
    ./ggbot.sh shell     进入 ggbot-app 容器的 shell
    ./ggbot.sh help      显示此帮助

  示例:
    ./ggbot.sh start                # 首次启动
    ./ggbot.sh cli                   # 终端对话
    ./ggbot.sh logs chromadb         # 看 chromadb 日志
    ./ggbot.sh test "退款要多久"      # 测试指定问题

  访问地址（启动后）:
    API:      http://localhost:8000
    Swagger:  http://localhost:8000/docs
    监控:     http://localhost:8000/monitor
    Prometheus: http://localhost:9090

EOF
}

# ── 交互菜单（无参数时）───────────────────────────────────────────────────────
show_menu() {
    cat << 'EOF'

  ╔═══════════════════════════════════════════════════════════════════╗
  ║          GGBot 智能客服系统                                        ║
  ╚═══════════════════════════════════════════════════════════════════╝

EOF
    echo -e "  ${BOLD}1)${NC} 启动服务          ${BOLD}2)${NC} CLI 对话           ${BOLD}3)${NC} 查看状态"
    echo -e "  ${BOLD}4)${NC} 查看日志          ${BOLD}5)${NC} 测试对话           ${BOLD}6)${NC} 健康检查"
    echo -e "  ${BOLD}7)${NC} 停止服务          ${BOLD}8)${NC} 重启服务           ${BOLD}9)${NC} 重新构建"
    echo -e "  ${BOLD}s)${NC} 进入容器 shell    ${BOLD}h)${NC} 帮助              ${BOLD}q)${NC} 退出"
    echo ""
    read -rp "请选择 [1-9/s/h/q]: " choice
    case "$choice" in
        1) cmd_start ;;
        2) cmd_cli ;;
        3) cmd_status ;;
        4) cmd_logs ;;
        5) cmd_test ;;
        6) cmd_health ;;
        7) cmd_stop ;;
        8) cmd_restart ;;
        9) cmd_build ;;
        s) cmd_shell ;;
        h) show_help ;;
        q) echo "再见 ʕ•ᴥ•ʔ" ;;
        *) warn "无效选择"; show_menu ;;
    esac
}

# ── 入口 ──────────────────────────────────────────────────────────────────────
case "${1:-menu}" in
    start|up)    cmd_start ;;
    stop)        cmd_stop ;;
    restart)     cmd_restart ;;
    status|ps)   cmd_status ;;
    logs)        shift; cmd_logs "$@" ;;
    cli|chat)    cmd_cli ;;
    test)        shift; cmd_test "$@" ;;
    health)      cmd_health ;;
    build)       cmd_build ;;
    down)        cmd_down ;;
    clean)        cmd_clean ;;
    shell)       cmd_shell ;;
    help|--help|-h) show_help ;;
    menu|"")     show_menu ;;
    *)           error "未知命令: $1"; show_help; exit 1 ;;
esac
