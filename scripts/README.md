# Scripts

## smoke_test.sh
```bash
chmod +x scripts/smoke_test.sh
scripts/smoke_test.sh /path/to/input.mp3
```

## test_client.py
```bash
pip install requests
python scripts/test_client.py /path/to/input.mp3
```

## ios_preview_request.sh
```bash
chmod +x scripts/ios_preview_request.sh
VOICE_ID=taylor_swift_singer TRIM_START=30 TRIM_DURATION=20 scripts/ios_preview_request.sh /path/to/input.mp3
```

## async_job_test.sh
```bash
chmod +x scripts/async_job_test.sh
scripts/async_job_test.sh /path/to/input.mp3
```
