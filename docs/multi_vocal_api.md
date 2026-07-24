# Sequential multi-vocal API (MVP)

The server supports two modes:

- `manual-sequential-v1`: a user-selected, non-overlapping sequence over the
  complete Demucs vocal stem.
- `medleyvox-two-singer-poc-v1`: two anonymous singer lanes estimated from the
  vocal stem. Assignments on different lanes may overlap.

MedleyVox lanes are estimates, not verified singer identities. They can leak or
swap across long-song chunk boundaries and must be auditioned by the user.

## 1. Inspect the source

`POST /ai-convert/vocal-analysis` with multipart field `source_audio` returns
the source duration and the only currently available source track:

```json
{
  "analysisVersion": "manual-sequential-v1",
  "durationSeconds": 53.42,
  "candidates": [{
    "trackId": "vocal_candidate_1",
    "label": "Separated vocal stem",
    "requiresManualSegmentation": true,
    "supportsOverlaps": false
  }]
}
```

When `analysisVersion` is `manual-sequential-v1`, the client must let the user
create non-overlapping time ranges on `vocal_candidate_1`. `trackId` is a
processing identifier, not a detected singer identity.

When `analysisVersion` is `medleyvox-two-singer-poc-v1`, the candidates are
`singer_1` and `singer_2`. The client should create one full-song assignment
for each lane, allow the user to choose a target voice per lane, and permit
overlap only between different lane IDs.

## 2. Render the selected ranges

`POST /ai-convert/multi-voice-conversion` accepts multipart fields:

| Field | Required | Description |
| --- | --- | --- |
| `source_audio` | yes | Original song audio |
| `vocalAssignments` | yes | JSON array described below |
| `output_format` | no | `mp3` (default), `wav`, or `m4a` |
| `mix_with_instrumental` | no | Include the Demucs instrumental; default `true` |
| `preserve_unassigned_vocals` | no | Keep unselected ranges from the original vocal stem; default `false` |

```json
[
  {
    "assignmentId": "verse-a",
    "sourceTrackId": "vocal_candidate_1",
    "voiceId": "ariana_grande",
    "segments": [{"start": 12.0, "end": 28.4}],
    "mixGain": 1.0,
    "pitchShift": 0
  },
  {
    "assignmentId": "verse-b",
    "sourceTrackId": "vocal_candidate_1",
    "voiceId": "the_weeknd",
    "segments": [{"start": 28.4, "end": 44.0}],
    "mixGain": 1.0
  }
]
```

Ranges must fall inside the source duration. They cannot overlap on the same
source track; overlap across `singer_1` and `singer_2` is supported. A `400`
response means the source/voice/range cannot be rendered; malformed JSON or
assignment schema returns `422`.

## Client flow

1. Call `vocal-analysis` after choosing a song.
2. For `singer_1` / `singer_2`, show two lanes and assign a catalog `voiceId`
   to each; for the fallback candidate, draw non-overlapping ranges.
4. Send the confirmed array to `multi-voice-conversion`.

For full-song replacement, assignments should cover every vocal range. Set
`preserve_unassigned_vocals=true` when the user explicitly wants unselected
original vocals retained. Simultaneous duet and backing-vocal support requires
a future speaker-separation model and is intentionally rejected in this MVP.
