import smtplib
from email.message import EmailMessage
import imaplib
EMAIL_ADDRESS = "cinemasnap4@gmail.com"
# APP_PASSWORD = "ohrg dqkc evmi mjtd"
APP_PASSWORD = "vuru sgxc ofgc dzzm"

try:
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_ADDRESS, APP_PASSWORD)
    print("✅ App Password is correct! Can access inbox.")
    mail.logout()
except imaplib.IMAP4.error:
    print("❌ App Password is INVALID or IMAP is disabled.")
except Exception as e:
    print("⚠️ Some other error occurred:", e)

