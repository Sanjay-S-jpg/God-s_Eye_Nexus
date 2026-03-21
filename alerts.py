# alerts.py - Alert System (Desktop + Telegram)
# For God's Eye Nexus - Crowd Intelligence System
#
# Supports: Desktop notifications, Telegram messages/photos, fight alerts
# Tokens loaded from .env via config.py for security

import os
import time
import threading
import requests
import cv2
import numpy as np
from datetime import datetime

# Import Telegram credentials from central config (loaded from .env)
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# ============================================================
# ALERT SETTINGS
# ============================================================
ALERT_COOLDOWN = 30  # Seconds between same type of alerts (prevent spam)
SAVE_ALERT_IMAGES = True  # Save images when alerts trigger

# ============================================================
# DESKTOP NOTIFICATIONS
# ============================================================

def show_desktop_notification(title, message, timeout=5):
    """
    Show a desktop popup notification.
    Works on Windows, Mac, and Linux.
    """
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Crowd Intelligence",
            timeout=timeout
        )
        print(f"[NOTIFICATION] {title}: {message}")
    except ImportError:
        print(f"[ALERT] {title}: {message}")
        print("[INFO] Install 'plyer' for desktop notifications: pip install plyer")
    except Exception as e:
        print(f"[NOTIFICATION ERROR] {e}")
        print(f"[ALERT] {title}: {message}")


# ============================================================
# TELEGRAM ALERTS
# ============================================================

def send_telegram_message(message):
    """
    Send a text message to Telegram.
    """
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_bot_token_here":
        print("[TELEGRAM] Bot not configured. Set TELEGRAM_BOT_TOKEN in .env file.")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            print(f"[TELEGRAM] Message sent successfully")
            return True
        else:
            print(f"[TELEGRAM ERROR] {response.text}")
            return False
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        return False


def send_telegram_photo(image, caption=""):
    """
    Send a photo to Telegram.
    image: numpy array (OpenCV frame)
    """
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_bot_token_here":
        print("[TELEGRAM] Bot not configured. Set TELEGRAM_BOT_TOKEN in .env file.")
        return False
    
    try:
        # Encode image to bytes
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()
        
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("alert.jpg", img_bytes, "image/jpeg")}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            print(f"[TELEGRAM] Photo sent successfully")
            return True
        else:
            print(f"[TELEGRAM ERROR] {response.text}")
            return False
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")
        return False


# ============================================================
# ALERT MANAGER
# ============================================================

class AlertManager:
    """
    Manages all alerts with cooldown to prevent spam.
    Sends both desktop notifications and Telegram messages.
    """
    
    def __init__(self, cooldown=30, enable_telegram=True, enable_desktop=True):
        """
        Args:
            cooldown: Seconds between same type alerts
            enable_telegram: Send Telegram messages
            enable_desktop: Show desktop popups
        """
        self.cooldown = cooldown
        self.enable_telegram = enable_telegram
        self.enable_desktop = enable_desktop
        
        # Track last alert time for each type
        self.last_alert_time = {}
        
        # Create alerts folder
        self.alerts_folder = "Alerts"
        if not os.path.exists(self.alerts_folder):
            os.makedirs(self.alerts_folder)
    
    def _can_alert(self, alert_type):
        """Check if enough time has passed since last alert of this type"""
        current_time = time.time()
        last_time = self.last_alert_time.get(alert_type, 0)
        
        if current_time - last_time >= self.cooldown:
            self.last_alert_time[alert_type] = current_time
            return True
        return False
    
    def _save_alert_image(self, frame, alert_type):
        """Save frame when alert triggers"""
        if frame is None:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.alerts_folder, f"{alert_type}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[SAVED] Alert image: {filename}")
        return filename
    
    def _send_async(self, func, *args):
        """Send alert in background thread (non-blocking)"""
        thread = threading.Thread(target=func, args=args, daemon=True)
        thread.start()
    
    # ============================================================
    # ALERT TRIGGERS
    # ============================================================
    
    def stampede_alert(self, risk_score, alert_level, frame=None):
        """
        Trigger stampede alert.
        Only alerts if risk is WARNING or CRITICAL.
        """
        if alert_level < 2:  # Only WARNING (2) or CRITICAL (3)
            return
        
        if not self._can_alert("stampede"):
            return
        
        level_names = ['SAFE', 'CAUTION', 'WARNING', 'CRITICAL']
        level_name = level_names[alert_level]
        
        title = f"⚠️ STAMPEDE {level_name}!"
        message = f"Risk Level: {risk_score}%\nImmediate attention required!"
        
        # Desktop notification
        if self.enable_desktop:
            show_desktop_notification(title, message, timeout=10)
        
        # Save image
        saved_path = None
        if SAVE_ALERT_IMAGES and frame is not None:
            saved_path = self._save_alert_image(frame, "stampede")
        
        # Telegram (async)
        if self.enable_telegram:
            telegram_msg = f"""
🚨 <b>STAMPEDE ALERT</b> 🚨

⚠️ Level: <b>{level_name}</b>
📊 Risk Score: <b>{risk_score}%</b>
🕐 Time: {datetime.now().strftime("%H:%M:%S")}

<i>Immediate attention required!</i>
"""
            if frame is not None:
                self._send_async(send_telegram_photo, frame, telegram_msg)
            else:
                self._send_async(send_telegram_message, telegram_msg)
    
    def weapon_alert(self, weapon_type="Weapon", frame=None):
        """
        Trigger weapon detection alert.
        """
        if not self._can_alert("weapon"):
            return
        
        title = "🔪 WEAPON DETECTED!"
        message = f"{weapon_type} detected in frame!\nAlert security immediately!"
        
        # Desktop notification
        if self.enable_desktop:
            show_desktop_notification(title, message, timeout=10)
        
        # Save image
        if SAVE_ALERT_IMAGES and frame is not None:
            self._save_alert_image(frame, "weapon")
        
        # Telegram (async)
        if self.enable_telegram:
            telegram_msg = f"""
🔪 <b>WEAPON DETECTED</b> 🔪

⚠️ Type: <b>{weapon_type}</b>
🕐 Time: {datetime.now().strftime("%H:%M:%S")}

<i>Alert security immediately!</i>
"""
            if frame is not None:
                self._send_async(send_telegram_photo, frame, telegram_msg)
            else:
                self._send_async(send_telegram_message, telegram_msg)
    
    def crowd_density_alert(self, count, threshold, frame=None):
        """
        Trigger high crowd density alert.
        """
        if count < threshold:
            return
        
        if not self._can_alert("density"):
            return
        
        title = "👥 HIGH CROWD DENSITY!"
        message = f"People count: {count}\nExceeds safe limit of {threshold}"
        
        # Desktop notification
        if self.enable_desktop:
            show_desktop_notification(title, message, timeout=8)
        
        # Save image
        if SAVE_ALERT_IMAGES and frame is not None:
            self._save_alert_image(frame, "density")
        
        # Telegram (async)
        if self.enable_telegram:
            telegram_msg = f"""
👥 <b>HIGH CROWD DENSITY</b>

📊 Current Count: <b>{count}</b>
⚠️ Safe Limit: <b>{threshold}</b>
🕐 Time: {datetime.now().strftime("%H:%M:%S")}

<i>Monitor situation closely!</i>
"""
            if frame is not None:
                self._send_async(send_telegram_photo, frame, telegram_msg)
            else:
                self._send_async(send_telegram_message, telegram_msg)
    
    def fight_alert(self, severity, person1_id, person2_id, frame=None):
        """
        Trigger fight/aggression detection alert.
        """
        if severity < 0.4:
            return
        
        if not self._can_alert("fight"):
            return
        
        level = "CRITICAL" if severity > 0.7 else "WARNING" if severity > 0.5 else "CAUTION"
        
        title = f"\U0001f44a FIGHT DETECTED - {level}!"
        message = f"Severity: {int(severity * 100)}%\nPersons: #{person1_id} vs #{person2_id}\nImmediate intervention required!"
        
        # Desktop notification
        if self.enable_desktop:
            show_desktop_notification(title, message, timeout=10)
        
        # Save image
        if SAVE_ALERT_IMAGES and frame is not None:
            self._save_alert_image(frame, "fight")
        
        # Telegram (async)
        if self.enable_telegram:
            telegram_msg = f"""
\U0001f44a <b>FIGHT DETECTED</b> \U0001f44a

\u26a0\ufe0f Level: <b>{level}</b>
\U0001f4ca Severity: <b>{int(severity * 100)}%</b>
\U0001f464 Persons: #{person1_id} vs #{person2_id}
\U0001f550 Time: {datetime.now().strftime("%H:%M:%S")}

<i>Immediate intervention required!</i>
"""
            if frame is not None:
                self._send_async(send_telegram_photo, frame, telegram_msg)
            else:
                self._send_async(send_telegram_message, telegram_msg)
    
    def custom_alert(self, title, message, alert_type="custom", frame=None):
        """
        Send a custom alert.
        """
        if not self._can_alert(alert_type):
            return
        
        # Desktop notification
        if self.enable_desktop:
            show_desktop_notification(title, message, timeout=8)
        
        # Save image
        if SAVE_ALERT_IMAGES and frame is not None:
            self._save_alert_image(frame, alert_type)
        
        # Telegram (async)
        if self.enable_telegram:
            telegram_msg = f"""
🔔 <b>{title}</b>

{message}
🕐 Time: {datetime.now().strftime("%H:%M:%S")}
"""
            if frame is not None:
                self._send_async(send_telegram_photo, frame, telegram_msg)
            else:
                self._send_async(send_telegram_message, telegram_msg)
    
    def test_alerts(self):
        """
        Test all alert channels.
        """
        print("\n" + "="*50)
        print("TESTING ALERT SYSTEM")
        print("="*50)
        
        # Test desktop
        print("\n[TEST] Desktop notification...")
        show_desktop_notification(
            "🧪 Test Alert",
            "Desktop notifications are working!",
            timeout=5
        )
        
        # Test Telegram
        print("\n[TEST] Telegram message...")
        success = send_telegram_message(
            "🧪 <b>Test Alert</b>\n\nTelegram integration is working!\n\n<i>Sent from Crowd Intelligence System</i>"
        )
        
        if success:
            print("[TEST] ✅ Telegram is configured correctly!")
        else:
            print("[TEST] ❌ Telegram failed. Check your BOT_TOKEN and CHAT_ID.")
        
        print("\n" + "="*50)
        print("ALERT TEST COMPLETE")
        print("="*50 + "\n")


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    print("Testing Alert System...")
    
    # Create manager
    manager = AlertManager(cooldown=5, enable_telegram=True, enable_desktop=True)
    
    # Run tests
    manager.test_alerts()