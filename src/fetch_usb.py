import os
import shutil
import time
from datetime import datetime

USB_PATH = "/media/trinhee/KINGSTON"
SAVE_DIR = "/home/trinhee/GM_automotive_challenge/object_model/test_usb"

if not os.path.exists(SAVE_DIR):
	os.makedirs(SAVE_DIR)

def fetch_from_usb():
	try:
		if not os.path.ismount(USB_PATH):
			print(f"USB not mounted at {USB_PATH}. Waiting...")
			return

		files = sorted([f for f in os.listdir(USB_PATH) if f.lower().endswith((".jpeg",".tiff"))])

		if not files:
			print("No image files found on USB.")
			return

		for file in files:
			usb_file_path = os.path.join(USB_PATH, file)
			save_path = os.path.join(SAVE_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{file}")

			shutil.copy2(usb_file_path, save_path)

			print(f"Copied: {usb_file_path} -> {save_path}")

	except Exception as e:
		print(f"Errror fetching from USB: {e}")

fetch_from_usb()
