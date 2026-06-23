# Beam Music iOS Integration Guide

## Usage mode
Beam Music iOS should use this server for **preview-first voice switching**.

## Recommended flow
1. User selects target voice
2. iOS sends current track audio with:
   - `voiceId`
   - `trim_start`
   - `trim_duration` (15~30s recommended)
3. Server returns converted preview audio
4. iOS swaps playback to converted preview
5. Optionally request full-track conversion in background later

## Suggested request params
- `voiceId`: selected voice
- `voiceType`: `singer` for singer presets
- `preserve_melody=true`
- `mix_with_instrumental=true`
- `output_format=mp3`
- `trim_start=<current playback time or phrase start>`
- `trim_duration=15~30`

## Why preview-first
Full song conversion per tap is too slow for responsive UX.
Short segment conversion is much better for:
- perceived realtime feel
- cost control
- cache hit rate
- retry behavior

## Cache key suggestion
`track_id + voice_id + trim_start + trim_duration + preserve_melody + pitch_shift`

## v1 recommendation
- preview conversion sync
- full-track conversion async later
- store converted preview in app cache and local file system
