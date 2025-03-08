# Setting Up `python-dotenv` and Discord Webhook for Logging

This guide explains how to securely store and use a Discord webhook URL in your Python project using `python-dotenv`. This ensures that sensitive information, like webhook URLs, is not hardcoded in your scripts.

---

## **1️⃣ Create a Discord Webhook**
To receive messages in Discord, you need to create a webhook:

1. Open Discord and go to your **server**.
2. Click the ⚙️ **Server Settings** (next to your server name).
3. In the left menu, go to **Integrations** → **Webhooks**.
4. Click **New Webhook**.
5. Name the webhook and select the **channel** where messages should be sent.
6. Click **Copy Webhook URL** – you’ll need this later.
7. Click **Save Changes**.

---

## **2️⃣ Install Dependencies**

Activate your existing Conda environment:
```bash
conda activate my_env  # Replace 'my_env' with your actual environment name
```

Then install `python-dotenv`:
```bash
pip install python-dotenv
```

---

## **3️⃣ Create a `.env` File**

Inside your project directory, create a new `.env` file:

```bash
touch .env
```

Open `.env` and add your Discord webhook URL:
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your-webhook-url
```

⚠️ **Important**: Never commit the `.env` file to Git. Add it to your `.gitignore` file:
```
# .gitignore
.env
```

---

## **4️⃣ Load Environment Variables in Python**

Modify your Python script to load the webhook URL securely:
```python
from dotenv import load_dotenv
import os
from discord_webhook import DiscordWebhook

# Load variables from .env file
load_dotenv()

# Retrieve the webhook URL
webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

if webhook_url:
    webhook = DiscordWebhook(url=webhook_url, content="Hello from Python!")
    webhook.execute()
else:
    print("🚨 Webhook URL not found! Make sure .env is set up correctly.")
```

---

## **5️⃣ Run Your Script**
Now, execute your Python script normally:
```bash
python your_script.py
```
If everything is set up correctly, you should see the message posted in your Discord channel.

---

## **6️⃣ Alternative: Set Webhook URL in Conda Environment (Optional)**
If you prefer, you can set the webhook URL directly in your Conda environment:
```bash
conda env config vars set DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-webhook-url"
```
Then, restart your terminal and activate the environment again:
```bash
conda activate my_env
```

You can now retrieve the variable in Python using `os.getenv("DISCORD_WEBHOOK_URL")`.

---

## **✅ Summary**
- Set up a Discord webhook in your server.
- Installed `python-dotenv` in your Conda environment.
- Stored the Discord webhook URL securely in a `.env` file.
- Loaded environment variables in Python.
- Prevented the `.env` file from being shared via `.gitignore`.
- (Optional) Used Conda environment variables for additional security.

Now, your Discord logging is securely integrated into your project! 🚀

