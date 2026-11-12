# Music App: notes for Claude

Personal tool for the owner (plays guitar, piano/keys, bass). Runs locally on their Windows PC
("WorkStation", NVIDIA GPU) as a web app at http://127.0.0.1:8000. See README.md for what it does.

## Run / test
- Install: `setup.bat` (Windows) or `./setup.sh`. Uses uv + Python 3.12 in `.venv`; PyTorch comes from
  the CUDA wheel index, and `basic-pitch` is installed with `--no-deps` on purpose (its deps pull TensorFlow;
  we use its ONNX model via onnxruntime).
- Start: `run.bat` / `./run.sh` (or `.venv\Scripts\python -m app`). Self-check: `python -m app.check`.
- Synthetic test song with an answer key: `python tools/make_test_song.py <dir>` (needs the `fluidsynth` CLI).
- Lint: `python -m pyflakes app tools`.

## Status (first version)
- Done and tested on CPU in the cloud: upload → 6 stems → minus track → tempo/key/chords → transcription →
  MusicXML/MIDI → web UI (mixer, chord chart with transpose/capo/Nashville, alphaTab sheet music with
  notation + tab for guitar/bass).
- Tested on the owner's PC (2026-11-12): `setup.bat`, CUDA on a GTX 1080 Ti (torch 2.14.1+cu126 still has
  sm_61), YouTube download without cookies, ~95 s per song, stem mixer drift < 20 ms.
- Fixed after the test song: notes transcribed from near-silent stem bleed (now gated to where the stem is
  within 30 dB of the mix, `separation.active_frames`); chords in songs without a bass guitar now use the
  low end of the keys/guitar for roots (bass weight 0.2). `fluidsynth` is not installed on the PC, so
  `tools/make_test_song.py` was not run.
- Owner's test song: https://www.youtube.com/watch?v=Nkdw4PyPvyI. Run it first on their PC and review the
  chord chart and sheet music with them.

## Accuracy benchmark (answer keys)
- `python tools/benchmark.py` renders 10 synthetic songs (random key/tempo/progression; piano, guitar, bass,
  drums from pretty_midi's TimGM6mb soundfont via `tinysoundfont`, installed in .venv but not in
  requirements.txt), runs the full pipeline, scores with mir_eval, logs to data/bench/history.jsonl.
  No singer: a soundfont "voice" isn't separated like a real one, so vocals/minus are checked on real songs.
- 2026-11-12 baseline -> tuning -> BS-RoFormer-SW stems: key 80->90->90%, tempo 80->90->90%,
  chords (majmin) 87.9->91.6->94.4%, notes F1 guitar 73.7->75.8->94.0, piano 63.9->66.8->84.5,
  bass 82.6->86.4->98.9. (data/bench_htdemucs keeps the htdemucs runs.)
- Stems: `config.stems_model()` = BS-RoFormer-SW (`bs-roformer-infer`, ~700 MB weights in
  ~/.cache/bs-roformer-infer, MIT architecture) on CUDA, htdemucs_6s on CPU. ~2x slower separation
  (owner's song: 118 s total). On the owner's song it finds no guitar at all (htdemucs had piano bleed in
  the guitar stem, which was the "guitar tab"): ask the owner whether the song has a guitar.
- Piano is now capped by Basic Pitch (~85%; thresholds re-swept, already optimal). Next: a piano model.
  Transkun is a candidate, but its deps pull a torchaudio that doesn't match torch 2.14 (try --no-deps).
- Remaining test misses: song02 key (C minor vs relative Eb major, guitar-only), song06 tempo (half time,
  no drums: deliberately left, doubling would also double real slow ballads).
- Tried and rejected: bass chroma capped at C3 (+1 pt on test songs but froze the owner's real song intro on
  A), dropping notes duplicated across stems (piano is often louder in the guitar stem, so it hurt),
  merging split notes (real repeats look the same). Always re-check the owner's song
  (job ae1b16456793) after a chord change, not only the benchmark.

## Next session (on the owner's PC: E:\Documents\Music App, WorkStation)
Owner's priority: get the core features working fully before adding anything new.
1. Run `setup.bat`; confirm `python -m app.check` reports the CUDA GPU and the models download.
2. Start `run.bat`, process the test song from YouTube (if YouTube asks to sign in, use
   `MUSICAPP_COOKIES_FROM_BROWSER` in settings.bat).
3. Review all three with the owner and fix what's wrong:
   - minus track cleanliness (vocal remnants? on GPU the minus track uses htdemucs_ft)
   - chord accuracy (compare against what they hear/play; tune `_QUALITY_BIAS`, the Viterbi
     self-transition 0.85 and the bass-root bonus in app/analysis.py)
   - guitar tab playability (fingering in app/notation.py `assign_frets`; extra notes from
     Basic Pitch thresholds in app/transcription.py `SPECS`)
4. Also check processing time on the GPU and that the stem mixer stays in sync.

## Ideas for later (the owner liked these)
Pitch/key shifting of the audio, synced lyrics with chords above them (Whisper), chord diagrams,
a better piano transcription model, separate acoustic vs electric guitar.

## Live link (owner wants to open the app from anywhere, just for themselves)
- Password protection is built in: set `MUSICAPP_PASSWORD` in settings.bat (app/auth.py; signed 30-day
  cookie, /static and /login stay public, API returns 401 without login).
- Owner first asked for Vercel; explained it can't run this (no GPU, short function limits), and processing
  must stay on the PC. Owner chose Cloudflare quick tunnel (2026-11-12): `share.bat` downloads cloudflared into
  .venv, refuses without a 12+ char password or if the port is already in use (an unprotected run.bat copy),
  then starts the app + tunnel. Refusal paths tested; the owner runs the tunnel itself (not tested end to end
  by Claude). If the changing link gets annoying, Tailscale Funnel gives a permanent one.
