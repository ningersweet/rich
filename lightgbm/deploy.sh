#!/bin/bash
# 比特币合约量化交易系统部署脚本

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}BTC 量化交易系统部署脚本${NC}"
echo -e "${GREEN}========================================${NC}\n"

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker 未安装，请先安装 Docker${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker 已安装${NC}"
}

# 检查 Docker Compose 是否安装
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Docker Compose 未安装，请先安装 Docker Compose${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker Compose 已安装${NC}"
}

# 检查配置文件
check_config() {
    if [ ! -f "config.yaml" ]; then
        echo -e "${RED}❌ config.yaml 不存在，请先创建配置文件${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 配置文件存在${NC}"
}

# 创建必要目录
create_dirs() {
    echo -e "\n${YELLOW}创建必要目录...${NC}"
    mkdir -p data models logs
    echo -e "${GREEN}✓ 目录创建完成${NC}"
}

# 构建 Docker 镜像
build_image() {
    echo -e "\n${YELLOW}构建 Docker 镜像...${NC}"
    docker build -t btc-quant:latest .
    echo -e "${GREEN}✓ 镜像构建完成${NC}"
}

# ========== 回测部署 ==========
deploy_backtest() {
    echo -e "\n${YELLOW}========== 部署回测服务 ==========${NC}"
    
    # 确保配置正确
    echo -e "${YELLOW}请确保 config.yaml 中已配置历史数据时间范围${NC}"
    
    # 运行回测容器
    docker-compose up backtest
    
    echo -e "${GREEN}✓ 回测完成，结果保存在 logs/ 目录${NC}"
    echo -e "${GREEN}✓ 模型保存在 models/ 目录${NC}"
}

# ========== 模拟盘部署 ==========
deploy_paper() {
    echo -e "\n${YELLOW}========== 部署模拟盘服务 ==========${NC}"
    
    # 检查模型是否存在
    if [ ! -f "models/model_latest.pkl" ]; then
        echo -e "${RED}❌ 模型文件不存在，请先运行回测生成模型${NC}"
        echo -e "${YELLOW}提示：运行 ./deploy.sh backtest${NC}"
        exit 1
    fi
    
    # 检查配置
    echo -e "${YELLOW}请确保 config.yaml 中：${NC}"
    echo -e "  - api.enable_trading: false"
    echo -e "  - api.key 和 api.secret 可留空（只读行情不需要签名）"
    read -p "配置是否正确？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}请先修改配置文件${NC}"
        exit 1
    fi
    
    # 启动模拟盘
    docker-compose up -d paper_trading
    
    echo -e "${GREEN}✓ 模拟盘服务已启动${NC}"
    echo -e "${YELLOW}查看日志：docker-compose logs -f paper_trading${NC}"
    echo -e "${YELLOW}停止服务：docker-compose stop paper_trading${NC}"
}

# ========== 实盘部署 ==========
deploy_live() {
    echo -e "\n${RED}========== 部署实盘服务（真实交易） ==========${NC}"
    
    # 检查模型是否存在
    if [ ! -f "models/model_latest.pkl" ]; then
        echo -e "${RED}❌ 模型文件不存在，请先运行回测生成模型${NC}"
        exit 1
    fi
    
    # 严格确认
    echo -e "${RED}⚠️  警告：即将启动实盘交易，会使用真实资金！${NC}"
    echo -e "${YELLOW}请确保 config.yaml 中已正确配置：${NC}"
    echo -e "  1. api.enable_trading: true"
    echo -e "  2. api.key: YOUR_REAL_API_KEY"
    echo -e "  3. api.secret: YOUR_REAL_API_SECRET"
    echo -e "  4. risk.* 风险参数已谨慎设置"
    echo -e "  5. 已在模拟盘充分测试"
    echo
    read -p "确认所有配置无误并继续？(yes/no) " -r
    echo
    if [[ ! $REPLY == "yes" ]]; then
        echo -e "${YELLOW}已取消实盘部署${NC}"
        exit 0
    fi
    
    # 二次确认
    echo -e "${RED}最后确认：真的要启动实盘交易吗？${NC}"
    read -p "输入 'START LIVE TRADING' 继续: " -r
    echo
    if [[ ! $REPLY == "START LIVE TRADING" ]]; then
        echo -e "${YELLOW}已取消实盘部署${NC}"
        exit 0
    fi
    
    # 启动实盘
    docker-compose up -d live_trading
    
    echo -e "${GREEN}✓ 实盘服务已启动${NC}"
    echo -e "${YELLOW}实时监控：docker-compose logs -f live_trading${NC}"
    echo -e "${YELLOW}停止交易：docker-compose stop live_trading${NC}"
    echo -e "${RED}⚠️  请定期检查日志和账户状态！${NC}"
}

# ========== 停止服务 ==========
stop_all() {
    echo -e "\n${YELLOW}停止所有服务...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ 所有服务已停止${NC}"
}

# ========== 查看日志 ==========
view_logs() {
    SERVICE=$1
    if [ -z "$SERVICE" ]; then
        echo -e "${YELLOW}可用服务：backtest, paper_trading, live_trading${NC}"
        read -p "请输入要查看日志的服务名: " SERVICE
    fi
    docker-compose logs -f $SERVICE
}

# ========== 清理 ==========
cleanup() {
    echo -e "\n${RED}⚠️  警告：这将删除所有容器、镜像和数据！${NC}"
    read -p "确认清理？(yes/no) " -r
    echo
    if [[ $REPLY == "yes" ]]; then
        docker-compose down -v
        docker rmi btc-quant:latest || true
        echo -e "${GREEN}✓ 清理完成${NC}"
    fi
}

# ========== 主菜单 ==========
main() {
    check_docker
    check_docker_compose
    check_config
    create_dirs
    
    if [ $# -eq 0 ]; then
        echo -e "\n${YELLOW}请选择操作：${NC}"
        echo "1) 构建镜像"
        echo "2) 回测（训练模型）"
        echo "3) 模拟盘（不真实下单）"
        echo "4) 实盘（真实交易）"
        echo "5) 停止所有服务"
        echo "6) 查看日志"
        echo "7) 清理所有"
        read -p "输入选项 (1-7): " choice
        
        case $choice in
            1) build_image ;;
            2) deploy_backtest ;;
            3) deploy_paper ;;
            4) deploy_live ;;
            5) stop_all ;;
            6) view_logs ;;
            7) cleanup ;;
            *) echo -e "${RED}无效选项${NC}" ;;
        esac
    else
        case $1 in
            build) build_image ;;
            backtest) deploy_backtest ;;
            paper) deploy_paper ;;
            live) deploy_live ;;
            stop) stop_all ;;
            logs) view_logs $2 ;;
            clean) cleanup ;;
            *)
                echo "用法: $0 {build|backtest|paper|live|stop|logs|clean}"
                echo ""
                echo "命令说明："
                echo "  build     - 构建 Docker 镜像"
                echo "  backtest  - 运行回测并训练模型"
                echo "  paper     - 启动模拟盘（不真实下单）"
                echo "  live      - 启动实盘（真实交易，谨慎使用）"
                echo "  stop      - 停止所有服务"
                echo "  logs      - 查看日志（需指定服务名）"
                echo "  clean     - 清理所有容器和镜像"
                exit 1
                ;;
        esac
    fi
}

main "$@"
