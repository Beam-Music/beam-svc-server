# Beam Music iOS Realtime Notes

## Important
For Beam Music iOS, this server is best treated as **near-real-time track conversion**, not low-latency live microphone conversion.

## Recommended product behavior
- user selects a voice on iOS
- app uploads current track audio or trimmed segment
- server converts 15~30s chunk or full selected preview
- app switches playback to converted result once ready

## Latency target
- 15s preview clip: 3~8s target on strong GPU
- 30s clip: 5~15s target
- full song: async job mode recommended

## iOS UX suggestions
- default preview conversion length: 15~30s
- prefetch conversion right after voice selection
- cache by `(track_id, voice_id, trim_start, trim_duration, parameters)`
- show converting state and allow background polling

## API suggestions for iOS use
Future additions:
- `trim_start`
- `trim_duration`
- `return_job=true`
- `GET /ai-convert/jobs/:jobId`
- `GET /ai-convert/results/:jobId`

## Not recommended yet
- true streaming live conversion over websocket
- per-frame online SVC for playback sync
- microphone real-time singing conversion in current architecture

## Good v1 scope
- fast preview conversion
- cached repeated requests
- async full-track conversion
