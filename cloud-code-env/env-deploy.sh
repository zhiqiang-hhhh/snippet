#!/bin/bash

# Claude Code Environment Setup Script
# Author: Claude Assistant
# Purpose: Automatically configure environment variables for Claude Code on macOS and Linux

set -e  # Exit on error

# Colors for output
RED='\033[1;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to display current environment variables
display_current_env() {
    print_info "当前环境变量状态："
    echo -e "${BLUE}----------------------------------------${NC}"
    
    # Get current shell type for proper variable display
    CURRENT_SHELL=$(basename "$SHELL")
    
    if [[ "$CURRENT_SHELL" == "fish" ]]; then
        # Fish shell syntax
        echo "ANTHROPIC_BASE_URL=$(set -q ANTHROPIC_BASE_URL && echo $ANTHROPIC_BASE_URL || echo '(未设置)')"
    else
        # Bash/Zsh syntax
        if [ -n "$ANTHROPIC_BASE_URL" ]; then
            echo "ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL"
        else
            echo "ANTHROPIC_BASE_URL=(未设置)"
        fi
    fi
    
    echo -e "${BLUE}----------------------------------------${NC}"
}

# Header
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Claude Code Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Display current environment variables before any changes
display_current_env
echo

# Check if API key is provided as argument
if [ $# -eq 0 ]; then
    print_error "请提供API密钥作为参数"
    echo "使用方法: $0 <your-api-key>"
    exit 1
fi

ANTHROPIC_API_KEY="$1"
ANTHROPIC_BASE_URL="http://127.0.0.1:3456"

# Detect OS and shell
detect_os_and_shell() {
    print_info "检测操作系统和Shell环境..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    else
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    
    # Detect Shell
    CURRENT_SHELL=$(basename "$SHELL")
    
    # Determine config file based on shell
    case "$CURRENT_SHELL" in
        bash)
            if [[ "$OS" == "macOS" ]]; then
                CONFIG_FILE="$HOME/.bash_profile"
            else
                CONFIG_FILE="$HOME/.bashrc"
            fi
            ;;
        zsh)
            CONFIG_FILE="$HOME/.zshrc"
            ;;
        fish)
            CONFIG_FILE="$HOME/.config/fish/config.fish"
            ;;
        *)
            print_error "不支持的Shell: $CURRENT_SHELL"
            exit 1
            ;;
    esac
    
    print_success "检测完成 - 系统: $OS, Shell: $CURRENT_SHELL"
    print_info "配置文件: $CONFIG_FILE"
}

# Function to add environment variables to config file
add_env_vars() {
    print_info "开始配置环境变量..."
    
    # Create backup
    if [ -f "$CONFIG_FILE" ]; then
        cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "已备份原配置文件"
    fi
    
    # Check if variables already exist
    if grep -q "ANTHROPIC_BASE_URL" "$CONFIG_FILE" 2>/dev/null; then
        print_warning "检测到已存在的Claude Code环境变量配置"
        print_info "正在清理所有现有配置..."
        
        # Remove ALL existing ANTHROPIC environment variable configurations
        # For bash/zsh: export VARIABLE=...
        # For fish: set -x VARIABLE ...
        if [[ "$CURRENT_SHELL" == "fish" ]]; then
            # Fish shell: remove 'set -x VARIABLE ...' patterns
            # Using -E for extended regex on macOS/BSD sed
            sed -i.tmp -E '/^[[:space:]]*set[[:space:]]+-x[[:space:]]+ANTHROPIC_BASE_URL/d' "$CONFIG_FILE" 2>/dev/null || true
        else
            # Bash/Zsh: remove 'export VARIABLE=...' patterns
            # Using -E for extended regex on macOS/BSD sed
            sed -i.tmp -E '/^[[:space:]]*export[[:space:]]+ANTHROPIC_BASE_URL=/d' "$CONFIG_FILE" 2>/dev/null || true
        fi
        
        # Also remove the marked sections for backward compatibility
        sed -i.tmp '/# Claude Code Environment Variables/,/# End Claude Code Environment Variables/d' "$CONFIG_FILE" 2>/dev/null || true
        
        # Clean up temporary files
        rm -f "$CONFIG_FILE.tmp"
        
        print_success "已彻底清理所有旧配置，准备写入新配置"
    fi
    
    # Add environment variables based on shell type
    if [[ "$CURRENT_SHELL" == "fish" ]]; then
        cat >> "$CONFIG_FILE" << EOF

# Claude Code Environment Variables
set -x ANTHROPIC_BASE_URL "$ANTHROPIC_BASE_URL"
# End Claude Code Environment Variables
EOF
    else
        cat >> "$CONFIG_FILE" << EOF

# Claude Code Environment Variables
export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
# End Claude Code Environment Variables
EOF
    fi
    
    print_success "环境变量已写入配置文件"
}

# Function to update .claude.json
update_claude_json() {
    print_info "更新 ~/.claude.json 配置..."
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        print_error "需要安装 jq 工具"
        if [[ "$OS" == "macOS" ]]; then
            print_info "请运行: brew install jq"
        else
            print_info "请运行: sudo apt-get install jq (Ubuntu/Debian) 或 sudo yum install jq (CentOS/RHEL)"
        fi
        return 1
    fi
    
    # Execute the jq command
    print_info "添加API密钥到Claude配置..."
    
    # Get the last 20 characters of the API key
    KEY_SUFFIX="${ANTHROPIC_API_KEY: -20}"
    
    # Create .claude.json if it doesn't exist
    if [ ! -f "$HOME/.claude.json" ]; then
        echo '{}' > "$HOME/.claude.json"
        print_info "创建新的 ~/.claude.json 文件"
    fi
    
    # Update the JSON file
    if (cat ~/.claude.json 2>/dev/null || echo 'null') | jq --arg key "$KEY_SUFFIX" '(. // {}) | .customApiKeyResponses.approved |= ([.[]?, $key] | unique)' > ~/.claude.json.tmp; then
        mv ~/.claude.json.tmp ~/.claude.json
        print_success "Claude配置已更新"
        
        # Display the updated customApiKeyResponses
        print_info "更新后的 customApiKeyResponses 内容:"
        echo -e "${BLUE}----------------------------------------${NC}"
        jq '.customApiKeyResponses' ~/.claude.json 2>/dev/null || echo "{}"
        echo -e "${BLUE}----------------------------------------${NC}"
    else
        print_error "更新Claude配置失败"
        rm -f ~/.claude.json.tmp
        return 1
    fi
}

# Function to source the config file
activate_config() {
    print_info "激活配置..."
    
    # Export variables for current session
    export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
    
    print_success "环境变量已在当前会话中激活"
    print_info "要在新的终端会话中使用，请运行以下命令："
    
    if [[ "$CURRENT_SHELL" == "fish" ]]; then
        echo -e "${GREEN}source $CONFIG_FILE${NC}"
    else
        echo -e "${GREEN}source $CONFIG_FILE${NC}"
    fi
    
    print_info "或者重新打开终端窗口"
}

# Function to verify configuration
verify_config() {
    print_info "验证配置..."
    
    # Check if variables are set
    if [ -n "$ANTHROPIC_BASE_URL" ]; then
        print_success "环境变量验证成功"
        echo "ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"
    else
        print_error "环境变量验证失败"
        return 1
    fi
    
    # Check .claude.json
    if [ -f "$HOME/.claude.json" ]; then
        if jq -e '.customApiKeyResponses.approved' "$HOME/.claude.json" &>/dev/null; then
            print_success "Claude配置文件验证成功"
        else
            print_warning "Claude配置文件存在但可能不完整"
        fi
    else
        print_error "Claude配置文件不存在"
    fi
}

# Function to configure Claude Code settings
configure_claude_code_settings() {
    print_info "配置 Claude Code settings.json..."
    
    # Create the .claude directory if it doesn't exist
    local claude_settings_dir="$HOME/.claude"
    if [ ! -d "$claude_settings_dir" ]; then
        mkdir -p "$claude_settings_dir"
        print_info "创建配置目录: $claude_settings_dir"
    fi
    
    local settings_file="$claude_settings_dir/settings.json"
    
    # Backup existing settings if they exist
    if [ -f "$settings_file" ]; then
        cp "$settings_file" "$settings_file.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "已备份现有的 Claude Code settings.json 文件"
    fi
    
    # Create the settings.json file with the required configuration
    cat > "$settings_file" << EOF
{
    "env": {
        "ANTHROPIC_API_KEY": "custom-api-key",
        "ANTHROPIC_BASE_URL": "$ANTHROPIC_BASE_URL"
    }
}
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Claude Code settings.json 已创建/更新: $settings_file"
        
        # Display the settings content
        print_info "当前 Claude Code settings.json 内容:"
        echo -e "${BLUE}----------------------------------------${NC}"
        cat "$settings_file" 2>/dev/null || echo "无法读取配置文件"
        echo -e "${BLUE}----------------------------------------${NC}"
    else
        print_error "创建 Claude Code settings.json 失败"
        return 1
    fi
}

# Function to configure Claude Code Router
configure_claude_code_router() {
    print_info "配置 Claude Code Router..."
    
    # Create the .claude-code-router directory if it doesn't exist
    local ccr_config_dir="$HOME/.claude-code-router"
    if [ ! -d "$ccr_config_dir" ]; then
        mkdir -p "$ccr_config_dir"
        print_info "创建配置目录: $ccr_config_dir"
    fi
    
    # Source config file path (in the same directory as this script)
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local source_config="$script_dir/config.json"
    local target_config="$ccr_config_dir/config.json"
    
    # Check if source config file exists
    if [ ! -f "$source_config" ]; then
        print_error "源配置文件不存在: $source_config"
        print_error "请先创建 config.json 文件并配置相关内容"
        return 1
    fi
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        print_error "需要安装 jq 工具来处理 JSON 配置文件"
        if [[ "$OS" == "macOS" ]]; then
            print_info "请运行: brew install jq"
        else
            print_info "请运行: sudo apt-get install jq (Ubuntu/Debian) 或 sudo yum install jq (CentOS/RHEL)"
        fi
        return 1
    fi
    
    # Validate that API key was provided
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        print_error "未提供 API key，无法配置 Claude Code Router"
        print_error "请在运行脚本时提供 API key 作为参数"
        return 1
    fi
    
    # Create a temporary config file with the API key replaced
    local temp_config="$source_config.tmp"
    if jq --arg api_key "$ANTHROPIC_API_KEY" '.Providers[0].api_key = $api_key' "$source_config" > "$temp_config"; then
        print_success "已将 API key 插入到配置中"
    else
        print_error "配置文件 JSON 格式错误或无法处理"
        rm -f "$temp_config"
        return 1
    fi
    
    # Backup existing config if it exists
    if [ -f "$target_config" ]; then
        cp "$target_config" "$target_config.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "已备份现有的 CCR 配置文件"
    fi
    
    # Copy the processed config file to CCR directory
    if cp "$temp_config" "$target_config"; then
        print_success "配置文件已复制到: $target_config"
        
        # Clean up temporary file
        rm -f "$temp_config"
        
        # Display the config content (mask the API key for security)
        print_info "当前 CCR 配置内容:"
        echo -e "${BLUE}----------------------------------------${NC}"
        if jq --arg masked_key "****${ANTHROPIC_API_KEY: -4}" '.Providers[0].api_key = $masked_key' "$target_config" 2>/dev/null; then
            true
        else
            echo "无法读取配置文件"
        fi
        echo -e "${BLUE}----------------------------------------${NC}"
    else
        print_error "配置文件复制失败"
        rm -f "$temp_config"
        return 1
    fi
}
install_npm_packages() {
    print_info "安装 Claude Code 相关包..."
    
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        print_error "npm 未找到，请先安装 Node.js 环境"
        print_info "可以运行 env-install.sh 脚本来安装 Node.js 环境"
        return 1
    fi
    
    print_info "当前 npm 版本: $(npm --version)"
    print_info "当前 Node.js 版本: $(node --version)"
    
    # Step 1: Install Claude Code first (required for Claude Code Router)
    print_info "第一步：安装 @anthropic-ai/claude-code..."
    if npm install -g @anthropic-ai/claude-code; then
        print_success "Claude Code 安装成功"
    else
        print_error "Claude Code 全局安装失败"
        print_info "尝试使用本地权限安装..."
        if npm install --prefix ~/.local @anthropic-ai/claude-code; then
            print_success "Claude Code 本地安装成功"
        else
            print_error "Claude Code 安装失败，请检查网络连接或权限"
            return 1
        fi
    fi
    
    # Step 2: Install Claude Code Router (requires Claude Code to be installed first)
    print_info "第二步：安装 @musistudio/claude-code-router..."
    if npm install -g @musistudio/claude-code-router; then
        print_success "Claude Code Router 安装成功"
    else
        print_error "Claude Code Router 全局安装失败"
        print_info "尝试使用本地权限安装..."
        if npm install --prefix ~/.local @musistudio/claude-code-router; then
            print_success "Claude Code Router 本地安装成功"
        else
            print_error "Claude Code Router 安装失败，请检查网络连接或权限"
            return 1
        fi
    fi
    
    # Verify installations
    print_info "验证安装结果..."
    
    # Check Claude Code
    if command -v claude-code &> /dev/null || [ -f "$HOME/.local/bin/claude-code" ]; then
        print_success "@anthropic-ai/claude-code 验证成功"
        # Try to get version if possible
        if command -v claude-code &> /dev/null; then
            local claude_version=$(claude-code --version 2>/dev/null || echo "版本信息不可用")
            print_info "Claude Code 版本: $claude_version"
        fi
    else
        print_warning "@anthropic-ai/claude-code 验证失败，可能需要重新加载终端"
    fi
    
    # Check Claude Code Router
    if command -v claude-code-router &> /dev/null || [ -f "$HOME/.local/bin/claude-code-router" ]; then
        print_success "@musistudio/claude-code-router 验证成功"
        # Try to get version if possible
        if command -v claude-code-router &> /dev/null; then
            local router_version=$(claude-code-router --version 2>/dev/null || echo "版本信息不可用")
            print_info "Claude Code Router 版本: $router_version"
        fi
    else
        print_warning "@musistudio/claude-code-router 验证失败，可能需要重新加载终端"
    fi
    
    print_success "npm 包安装完成"
}

# Main execution
main() {
    # Step 1: Detect OS and Shell
    detect_os_and_shell
    echo
    
    # Step 2: Add environment variables
    add_env_vars
    echo
    
    # Step 3: Update .claude.json
    update_claude_json
    echo
    
    # Step 4: Install npm packages
    install_npm_packages
    echo
    
    # Step 5: Configure Claude Code settings
    configure_claude_code_settings
    echo
    
    # Step 6: Configure Claude Code Router
    configure_claude_code_router
    echo
    
    # Step 7: Activate configuration
    activate_config
    echo
    
    # Step 8: Verify configuration
    verify_config
    echo
    
    print_success "Claude Code环境配置完成！"
    echo -e "${BLUE}========================================${NC}"

    # Display installed packages info
    echo -e "${BLUE}已安装的包：${NC}"
    echo -e "📦 @anthropic-ai/claude-code - Claude Code CLI 工具"
    echo -e "📦 @musistudio/claude-code-router - Claude Code 路由器"
    echo ""
    
    # Important reminder in red
    echo
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                          ║${NC}"
    echo -e "${RED}║    请关闭终端后重新打开，开始 claude code 使用～        ║${NC}"
    echo -e "${RED}║                                                          ║${NC}"
    echo -e "${RED}║    可以运行 'claude-code --help' 查看使用帮助           ║${NC}"
    echo -e "${RED}║                                                          ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Run main function
main

# Exit successfully
exit 0