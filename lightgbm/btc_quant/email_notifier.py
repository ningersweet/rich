"""
邮件通知模块
用于交易时发送邮件提醒
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging


class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self, smtp_host, smtp_port, sender_email, sender_password, receiver_email, enabled=True):
        """
        初始化邮件通知器
        
        Args:
            smtp_host: SMTP服务器地址
            smtp_port: SMTP端口
            sender_email: 发件人邮箱
            sender_password: 发件人密码/授权码
            receiver_email: 收件人邮箱
            enabled: 是否启用邮件通知
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"✉️  邮件通知已启用，发送至: {self.receiver_email}")
        else:
            self.logger.info("✉️  邮件通知已禁用")
    
    def send_email(self, subject, body):
        """
        发送邮件
        
        Args:
            subject: 邮件主题
            body: 邮件正文
        """
        if not self.enabled:
            return
        
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = subject
            
            # 添加正文
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 连接SMTP服务器（添加超时）
            self.logger.debug(f"正在连接SMTP服务器 {self.smtp_host}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                self.logger.debug("启动TLS加密")
                server.starttls()  # 启用TLS加密
                
                self.logger.debug("验证登录")
                server.login(self.sender_email, self.sender_password)
                
                self.logger.debug("发送邮件")
                server.send_message(msg)
            
            self.logger.info(f"📧 邮件发送成功: {subject}")
            
        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"❌ 邮件认证失败: {e}")
            self.logger.error("请检查邮箱密码/授权码是否正确")
        except smtplib.SMTPException as e:
            self.logger.error(f"❌ SMTP错误: {e}")
        except Exception as e:
            self.logger.error(f"❌ 邮件发送失败: {e}")
    
    def notify_open_position(self, side, quantity, price, exposure, rr, prob, balance):
        """
        开仓通知
        
        Args:
            side: 方向 (long/short)
            quantity: 数量
            price: 价格
            exposure: 敞口
            rr: 盈亏比
            prob: 置信度
            balance: 当前余额
        """
        if not self.enabled:
            return
        
        direction_cn = "做多" if side == "long" else "做空"
        
        subject = f"🔔 开仓通知 - {direction_cn} BTC"
        
        body = f"""
【开仓通知】

交易方向: {direction_cn}
开仓时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
开仓价格: {price:,.2f} USDT
开仓数量: {quantity:.4f} BTC
当前余额: {balance:,.2f} USDT

信号质量:
- 盈亏比: {rr:.2f}
- 置信度: {prob:.3f}
- 敞口倍数: {exposure:.2f}x

--
BTC量化交易系统
"""
        
        self.send_email(subject, body)
    
    def notify_close_position(self, side, quantity, entry_price, exit_price, pnl, pnl_pct, reason, balance):
        """
        平仓通知
        
        Args:
            side: 方向 (long/short)
            quantity: 数量
            entry_price: 开仓价格
            exit_price: 平仓价格
            pnl: 盈亏金额
            pnl_pct: 盈亏百分比
            reason: 平仓原因
            balance: 当前余额
        """
        if not self.enabled:
            return
        
        direction_cn = "做多" if side == "long" else "做空"
        is_profit = pnl > 0
        emoji = "🟢 盈利" if is_profit else "🔴 亏损"
        
        subject = f"{emoji} 平仓通知 - {direction_cn} BTC"
        
        body = f"""
【平仓通知】

交易方向: {direction_cn}
平仓时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
平仓原因: {reason}

价格信息:
- 开仓价格: {entry_price:,.2f} USDT
- 平仓价格: {exit_price:,.2f} USDT
- 价格变化: {(exit_price - entry_price):,.2f} USDT

盈亏信息:
- 盈亏金额: {pnl:,.2f} USDT
- 盈亏比例: {pnl_pct:.2f}%
- 当前余额: {balance:,.2f} USDT

--
BTC量化交易系统
"""
        
        self.send_email(subject, body)
    
    def notify_risk_alert(self, alert_type, message, current_drawdown=None, balance=None):
        """
        风控警告通知
        
        Args:
            alert_type: 警告类型
            message: 警告信息
            current_drawdown: 当前回撤
            balance: 当前余额
        """
        if not self.enabled:
            return
        
        subject = f"⚠️  风控警告 - {alert_type}"
        
        body = f"""
【风控警告】

警告类型: {alert_type}
警告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
警告信息: {message}
"""
        
        if current_drawdown is not None:
            body += f"\n当前回撤: {current_drawdown:.2f}%"
        
        if balance is not None:
            body += f"\n当前余额: {balance:,.2f} USDT"
        
        body += "\n\n--\nBTC量化交易系统"
        
        self.send_email(subject, body)
