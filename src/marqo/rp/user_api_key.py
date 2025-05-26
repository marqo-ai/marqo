import threading
import time
import boto3
import os
from fastapi import Request, HTTPException

API_KEY_CACHE = set()
REFRESH_INTERVAL_SECS = 30


def start_api_key_refresh_thread():
    thread = threading.Thread(target=_refresh_loop, daemon=True)
    thread.start()


def _refresh_loop():
    while True:
        try:
            _refresh_keys()
        except Exception as e:
            print(f"[Auth] Error refreshing keys: {e}")
        time.sleep(REFRESH_INTERVAL_SECS)


def _refresh_keys():
    system_account_id = os.environ.get("SYS_ACC_ID", "gje7jbgi")
    table_name = f"{system_account_id}_API_keys"
    region = os.environ.get("APPLICATION_AWS_REGION", "us-east-1")
    ddb = boto3.client("dynamodb", region_name=region)

    keys = set()
    scan_kwargs = {
        "TableName": table_name,
        "ProjectionExpression": "encrypted_key",
    }

    resp = ddb.scan(**scan_kwargs)
    items = resp.get("Items", [])
    while True:
        for item in items:
            k = item.get("encrypted_key", {}).get("S")
            if k:
                keys.add(k)
        if "LastEvaluatedKey" in resp:
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            resp = ddb.scan(**scan_kwargs)
            items = resp.get("Items", [])
        else:
            break

    API_KEY_CACHE.clear()
    API_KEY_CACHE.update(keys)
    print(f"[Auth] Refreshed {len(keys)} keys")


def user_api_key_auth(request: Request):
    key = request.headers.get("x-api-key", "")
    if key not in API_KEY_CACHE:
        print("[Auth] API key is invalid")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")
    print("[Auth] API key is valid")
    return key
