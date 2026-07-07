from pathlib import Path
import sys
import requests

BASE_URL = "http://127.0.0.1:8081"
VOICE_ID = "dionn_v1_singing"


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/test_client.py /path/to/input.mp3")
        raise SystemExit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"missing file: {input_path}")
        raise SystemExit(1)

    print(requests.get(f"{BASE_URL}/ai-convert/health", timeout=10).json())
    print(requests.get(f"{BASE_URL}/ai-convert/voices", timeout=10).json())

    with input_path.open("rb") as f:
        response = requests.post(
            f"{BASE_URL}/ai-convert/voice-conversion",
            files={"source_audio": (input_path.name, f, "audio/mpeg")},
            data={
                "voiceId": VOICE_ID,
                "voiceType": "default",
                "language": "en",
                "preserve_melody": "true",
                "mix_with_instrumental": "true",
                "output_format": "mp3",
            },
            timeout=600,
        )

    if response.headers.get("content-type", "").startswith("application/json"):
        print(response.status_code, response.json())
        raise SystemExit(1)

    output_path = Path("/tmp/beam_svc_test_client.mp3")
    output_path.write_bytes(response.content)
    print(f"saved: {output_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
