from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import VocalAssignment
from app.services.audio_utils import convert_to_wav, encode_audio, probe_duration_seconds, run_command
from app.services.medleyvox_service import MedleyVoxService
from app.services.pipeline import BeamSVCPipeline, ConversionOptions
from app.settings import settings


@dataclass
class MultiConversionOptions:
    output_format: str
    mix_with_instrumental: bool = True
    preserve_unassigned_vocals: bool = False


class MultiVocalPipeline:
    """Sequential-duet renderer built on the existing Demucs/RVC pipeline.

    `vocal_candidate_1` means the complete Demucs vocal stem. `singer_1` and
    `singer_2` become available only when MedleyVox is configured.
    """

    candidate_id = "vocal_candidate_1"

    def __init__(self) -> None:
        self.single = BeamSVCPipeline()
        self.medleyvox = MedleyVoxService()

    def analyze(self, input_path: Path, workdir: Path) -> float:
        normalized = convert_to_wav(input_path, workdir / "normalized.wav")
        return probe_duration_seconds(normalized)

    def convert(
        self,
        input_path: Path,
        assignments: list[VocalAssignment],
        options: MultiConversionOptions,
    ) -> bytes:
        workdir = input_path.parent
        normalized = convert_to_wav(input_path, workdir / "normalized.wav")
        duration = probe_duration_seconds(normalized)
        self._validate_assignments(assignments, duration)
        vocals, instrumental = self.single.demucs.separate(normalized, workdir)
        source_tracks = {self.candidate_id: vocals}
        if any(assignment.sourceTrackId in self.medleyvox.source_track_ids for assignment in assignments):
            source_tracks.update(self.medleyvox.separate(vocals, workdir))

        rendered_stems: list[tuple[Path, float]] = []
        for assignment in assignments:
            voice = self.single.registry.get_voice(assignment.voiceId)
            if voice is None:
                raise ValueError(f"Unknown voiceId: {assignment.voiceId}")
            for segment_index, segment in enumerate(assignment.segments, start=1):
                segment_dir = workdir / "assignments" / assignment.assignmentId / f"segment_{segment_index:03d}"
                segment_dir.mkdir(parents=True, exist_ok=True)
                source_segment = self._extract_segment(
                    source_tracks[assignment.sourceTrackId],
                    segment_dir / "source.wav",
                    segment.start,
                    segment.end,
                )
                conversion_options = ConversionOptions(
                    voice_id=assignment.voiceId,
                    voice_type=None,
                    language="und",
                    preserve_melody=True,
                    mix_with_instrumental=False,
                    output_format="wav",
                    pitch_shift=assignment.pitchShift if assignment.pitchShift is not None else (voice.model.defaultPitchShift if voice.model else 0),
                )
                f0_path = self.single.rmvpe.extract_f0(source_segment, segment_dir)
                converted = self.single._infer_vocals_chunked(
                    vocals_wav=source_segment,
                    f0_path=f0_path,
                    voice=voice,
                    workdir=segment_dir,
                    options=conversion_options,
                )
                placed = self._place_on_timeline(
                    converted,
                    segment_dir / "placed.wav",
                    start=segment.start,
                    duration=duration,
                )
                rendered_stems.append((placed, assignment.mixGain))

        final_wav = workdir / "final_multi_mix.wav"
        inputs: list[tuple[Path, float]] = [(instrumental, 1.0)] if options.mix_with_instrumental else []
        if options.preserve_unassigned_vocals:
            inputs.append((self._mute_assigned_ranges(vocals, workdir / "unassigned_vocals.wav", assignments), 1.0))
        inputs.extend(rendered_stems)
        if not inputs:
            raise ValueError("At least one output stem is required")
        self._mix(inputs, final_wav)
        output = workdir / f"output.{options.output_format}"
        encode_audio(final_wav, output, options.output_format)
        return output.read_bytes()

    def _validate_assignments(self, assignments: list[VocalAssignment], duration: float) -> None:
        if not assignments:
            raise ValueError("At least one vocal assignment is required")
        ranges: list[tuple[float, float, str, str]] = []
        for assignment in assignments:
            if assignment.sourceTrackId not in {self.candidate_id, *self.medleyvox.source_track_ids}:
                raise ValueError(f"Unsupported sourceTrackId: {assignment.sourceTrackId}")
            if assignment.sourceTrackId in self.medleyvox.source_track_ids and not self.medleyvox.available():
                raise ValueError("Multi-singer separation is not available on this server")
            for segment in assignment.segments:
                if segment.end <= segment.start:
                    raise ValueError("Each segment end must be greater than start")
                if segment.end > duration + 0.05:
                    raise ValueError("A segment exceeds the source duration")
                ranges.append((segment.start, segment.end, assignment.assignmentId, assignment.sourceTrackId))
        ranges.sort()
        for (_, previous_end, previous_id, previous_track), (start, _, current_id, current_track) in zip(ranges, ranges[1:]):
            if start < previous_end - 0.01 and previous_track == current_track:
                raise ValueError(f"Overlapping segments are not supported ({previous_id}, {current_id})")

    def _extract_segment(self, source: Path, output: Path, start: float, end: float) -> Path:
        run_command([settings.ffmpeg_command, "-y", "-ss", str(start), "-i", str(source), "-t", str(end - start), "-ar", "48000", "-ac", "2", str(output)])
        return output

    def _place_on_timeline(self, source: Path, output: Path, start: float, duration: float) -> Path:
        delay_ms = max(0, round(start * 1000))
        fade = min(0.03, max(0.0, duration - start) / 2)
        filter_parts = [f"adelay={delay_ms}|{delay_ms}"]
        if fade:
            filter_parts.append(f"afade=t=in:st={start}:d={fade}")
        run_command([settings.ffmpeg_command, "-y", "-i", str(source), "-af", ",".join(filter_parts), "-t", str(duration), "-ar", "48000", "-ac", "2", str(output)])
        return output

    def _mute_assigned_ranges(self, source: Path, output: Path, assignments: list[VocalAssignment]) -> Path:
        ranges = sorted((segment.start, segment.end) for assignment in assignments for segment in assignment.segments)
        expressions = [f"between(t,{start},{end})" for start, end in ranges]
        expression = "+".join(expressions) or "0"
        run_command([settings.ffmpeg_command, "-y", "-i", str(source), "-af", f"volume='if(gt({expression},0),0,1)':eval=frame", "-ar", "48000", "-ac", "2", str(output)])
        return output

    def _mix(self, inputs: list[tuple[Path, float]], output: Path) -> Path:
        command = [settings.ffmpeg_command, "-y"]
        for path, _ in inputs:
            command.extend(["-i", str(path)])
        filters = [f"[{index}:a]volume={gain}[a{index}]" for index, (_, gain) in enumerate(inputs)]
        labels = "".join(f"[a{index}]" for index in range(len(inputs)))
        filters.append(f"{labels}amix=inputs={len(inputs)}:duration=longest:normalize=0,alimiter=limit=0.98[out]")
        command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "48000", "-ac", "2", str(output)])
        run_command(command)
        return output
