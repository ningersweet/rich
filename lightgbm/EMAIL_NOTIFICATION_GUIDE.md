# 邮件通知配置指南

## 📧 功能说明

邮件通知功能会在以下情况发送邮件提醒：

1. **开仓通知** - 新开仓位时发送，包含方向、价格、敞口等信息
2. **平仓通知** - 平仓时发送，包含盈亏、平仓原因等信息
3. **风控警告** - 触发风控暂停时发送（回撤暂停、每日亏损限制）

---

## ⚙️ 配置步骤

### 1. 复制配置模板

```bash
cp email_config.yaml.example email_config.yaml
```

### 2. 编辑配置文件

打开 `email_config.yaml`，修改以下配置：

```yaml
email:
  enabled: true  # 设置为true启用邮件通知
  
  # SMTP服务器配置
  smtp_host: "smtp.gmail.com"  # 根据你的邮箱提供商修改
  smtp_port: 587
  
  # 发件人信息
  sender_email: "your_email@gmail.com"  # 你的邮箱
  sender_password: "your_app_password"  # 邮箱授权码（不是登录密码！）
  
  # 收件人信息
  receiver_email: "your_email@gmail.com"  # 接收通知的邮箱
```

---

## 📮 常用邮箱配置

### Gmail

```yaml
smtp_host: "smtp.gmail.com"
smtp_port: 587
```

**获取授权码**：
1. 访问 https://myaccount.google.com/security
2. 启用"两步验证"
3. 生成"应用专用密码"
4. 使用生成的16位密码作为 `sender_password`

### QQ邮箱

```yaml
smtp_host: "smtp.qq.com"
smtp_port: 587
```

**获取授权码**：
1. 登录QQ邮箱网页版
2. 设置 → 账户 → 开启SMTP服务
3. 生成授权码
4. 使用授权码作为 `sender_password`

### 163邮箱

```yaml
smtp_host: "smtp.163.com"
smtp_port: 587
```

**获取授权码**：
1. 登录163邮箱
2. 设置 → POP3/SMTP/IMAP
3. 开启SMTP服务并生成授权码
4. 使用授权码作为 `sender_password`

---

## 🔗 集成到主配置

将邮件配置添加到 `config.yaml`：

```yaml
# ... 其他配置 ...

# 邮件通知配置
email:
  enabled: true
  smtp_host: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your_email@gmail.com"
  sender_password: "your_app_password"
  receiver_email: "your_email@gmail.com"

# 通知配置
notifications:
  notify_on_open: true
  notify_on_close: true
  notify_on_risk_alert: true
```

---

## 📧 邮件示例

### 开仓通知

```
【开仓通知】

交易方向: 做多
开仓时间: 2026-02-20 23:30:00
开仓价格: 67,500.00 USDT
开仓数量: 0.1500 BTC
当前余额: 5,000.00 USDT

信号质量:
- 盈亏比: 3.50
- 置信度: 0.850
- 敞口倍数: 9.50x

--
BTC量化交易系统
```

### 平仓通知

```
【平仓通知】

交易方向: 做多
平仓时间: 2026-02-20 23:45:00
平仓原因: 持仓周期(17/17)K线

价格信息:
- 开仓价格: 67,500.00 USDT
- 平仓价格: 68,200.00 USDT
- 价格变化: 700.00 USDT

盈亏信息:
- 盈亏金额: 45.50 USDT
- 盈亏比例: 1.04%
- 当前余额: 5,045.50 USDT

--
BTC量化交易系统
```

### 风控警告

```
【风控警告】

警告类型: 回撤暂停
警告时间: 2026-02-20 23:50:00
警告信息: 回撤达到 10.50%，已暂停交易至明日

当前回撤: 10.50%
当前余额: 4,475.00 USDT

--
BTC量化交易系统
```

---

## ⚠️ 注意事项

1. **使用授权码而非密码**：大部分邮箱服务商要求使用应用专用授权码，不能使用登录密码
2. **安全存储**：`email_config.yaml` 和 `config.yaml` 已添加到 `.gitignore`，不会被提交到Git
3. **测试发送**：首次配置后建议先测试发送是否成功
4. **邮件频率**：邮件仅在关键事件（开仓/平仓/风控）时发送，不会频繁骚扰
5. **网络要求**：确保运行环境能访问SMTP服务器（部分服务器可能需要设置防火墙）

---

## 🧪 测试邮件功能

可以编写一个简单的测试脚本：

```python
from btc_quant.config import load_config
from btc_quant.email_notifier import EmailNotifier

# 加载配置
cfg = load_config('config.yaml')

# 初始化邮件通知器
email_cfg = cfg.get('email', {})
notifier = EmailNotifier(
    smtp_host=email_cfg['smtp_host'],
    smtp_port=email_cfg['smtp_port'],
    sender_email=email_cfg['sender_email'],
    sender_password=email_cfg['sender_password'],
    receiver_email=email_cfg['receiver_email'],
    enabled=True
)

# 发送测试邮件
notifier.send_email(
    subject="📧 BTC量化系统 - 邮件测试",
    body="这是一封测试邮件，邮件通知功能配置成功！"
)

print("测试邮件已发送，请检查收件箱")
```

---

## 🔧 故障排查

### 邮件发送失败

1. **检查授权码**：确保使用的是应用专用授权码，不是登录密码
2. **检查SMTP配置**：确认smtp_host和smtp_port正确
3. **检查网络连接**：确保能访问SMTP服务器（`telnet smtp.gmail.com 587`）
4. **查看日志**：检查 `logs/btc_quant.log` 中的错误信息

### Gmail报错"less secure app"

- Gmail已禁用"不够安全的应用"登录
- 必须启用两步验证并使用"应用专用密码"

### QQ邮箱连接超时

- 确保已开启SMTP服务
- 检查是否被防火墙拦截
- 尝试使用SSL端口465

---

## 📝 更新日志

- **2026-02-20**: 初始版本发布
  - 支持开仓/平仓/风控邮件通知
  - 支持Gmail/QQ/163等常用邮箱
  - 集成到run_live_dynamic_exposure.py
