import os
import uuid
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ---------------- CONFIG ---------------- #
N_SESSIONS = 500_000

np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# ---------------------------------------- #

OUT_DIR = os.path.join("data", "synthetic")
OUT_PATH = os.path.join(OUT_DIR, "sessions.parquet")


def create_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def generate_users(n_users=50_000):
    users = []

    for _ in range(n_users):
        users.append({
            "user_id": str(uuid.uuid4()),
            "base_device_id": str(uuid.uuid4()),
            "base_country": fake.country_code(),
            "base_os": random.choice(["Windows", "macOS", "Linux", "Android", "iOS"]),
            "base_browser": random.choice(["Chrome", "Firefox", "Edge", "Safari"])
        })

    return users


def sample_session(user, base_time):

    is_fraud = np.random.rand() < 0.12

    start_time = base_time + timedelta(
        minutes=np.random.randint(-43200, 43200)
    )

    if is_fraud and np.random.rand() < 0.6:
        device_id = str(uuid.uuid4())
        device_mismatch = 1
    else:
        device_id = user["base_device_id"]
        device_mismatch = 0

    if is_fraud and np.random.rand() < 0.4:
        country = fake.country_code()
    else:
        country = user["base_country"]

    if is_fraud and np.random.rand() < 0.4:
        os_name = random.choice(["Windows", "macOS", "Linux", "Android", "iOS"])
        browser = random.choice(["Chrome", "Firefox", "Edge", "Safari"])
    else:
        os_name = user["base_os"]
        browser = user["base_browser"]

    vpn_flag = 1 if (is_fraud and np.random.rand() < 0.55) or \
                    (not is_fraud and np.random.rand() < 0.10) else 0

    if is_fraud:
        mouse_speed_mean = np.random.normal(600, 200)
        mouse_speed_std = abs(np.random.normal(220, 80))
    else:
        mouse_speed_mean = np.random.normal(350, 100)
        mouse_speed_std = abs(np.random.normal(120, 50))

    mouse_speed_mean = max(20, min(mouse_speed_mean, 1500))

    if is_fraud:
        key_latency_mean = np.random.normal(90, 30)
        key_latency_std = abs(np.random.normal(80, 30))
    else:
        key_latency_mean = np.random.normal(160, 50)
        key_latency_std = abs(np.random.normal(60, 20))

    key_latency_mean = max(30, min(key_latency_mean, 1000))

    if is_fraud:
        scroll_total = abs(np.random.normal(1000, 600))
    else:
        scroll_total = abs(np.random.normal(3500, 1500))

    ip_entropy = np.random.uniform(0.6, 1.0) if is_fraud else np.random.uniform(0.1, 0.7)

    n_events = int(max(5, np.random.normal(25, 10))) if is_fraud \
               else int(max(10, np.random.normal(60, 20)))

    return {
        "session_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "start_time": start_time,

        "mouse_speed_mean": float(mouse_speed_mean),
        "mouse_speed_std": float(mouse_speed_std),

        "keystroke_latency_mean": float(key_latency_mean),
        "keystroke_latency_std": float(key_latency_std),

        "scroll_delta_total": float(scroll_total),

        "ip_entropy": float(ip_entropy),

        "n_events": n_events,

        "device_id": device_id,
        "browser": browser,
        "os": os_name,
        "country": country,

        "vpn_flag": vpn_flag,
        "device_mismatch_flag": device_mismatch,

        "label": int(is_fraud)
    }


def main():
    create_dirs()

    users = generate_users()
    base_time = datetime.utcnow()
    rows = []

    print("Generating synthetic sessions...")

    for i in range(N_SESSIONS):

        if i > 0 and i % 50_000 == 0:
            print(f"{i} sessions generated")

        user = random.choice(users)
        rows.append(sample_session(user, base_time))

    df = pd.DataFrame(rows)

    df.to_parquet(OUT_PATH, index=False)

    print("\n✅ Completed!")
    print("Saved →", OUT_PATH)
    print("\nFraud ratio:")
    print(df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
