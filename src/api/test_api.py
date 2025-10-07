import requests
import time
import random
import json

FEATURE_NAMES = [
    "Global_active_power", "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "Sub_total", "Apparent_power", "Power_factor", "month", "hour", "day_of_week"
]

def generate_features(auto=True):
    """Generate a random or fixed feature vector."""
    if auto:
        return [
            1.334, 5.6, 0.0, 0.0, 0.0,
            0.0, 1.360128, 0.98079004,
            random.randint(0, 11), random.randint(0, 23), random.randint(0, 6)
        ]
    else:
        # fixed reference input
        return [1.334, 5.6, 0, 0, 0, 0, 1.360128, 0.98079004, 1, 22, 3]


def send_request(endpoint: str, features):
    """Send POST request to FastAPI endpoint."""
    url = f"http://127.0.0.1:8000/{endpoint}"
    payload = {"features": features}

    for attempt in range(5):
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                print(f"\n✅ Response from /{endpoint}:")
                print(json.dumps(resp.json(), indent=2))
                return
            else:
                print(f"⚠️ Server returned status {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"⏳ Waiting for FastAPI server... (attempt {attempt+1}/5)")
            time.sleep(2)
    print("❌ Could not connect to FastAPI server. Make sure it's running.")


def main(args):
    features = generate_features(auto=True)
    if args.endpoint in ["predict", "explain"]:
        send_request(args.endpoint, features)
    elif args.endpoint == "both":
        send_request("predict", features)
        time.sleep(1)
        send_request("explain", features)
    else:
        print("Invalid endpoint. Choose 'predict', 'explain', or 'both'.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--endpoint", default="predict", help="choose 'predict', 'explain', or 'both'")
    args = parser.parse_args()
    main(args)




