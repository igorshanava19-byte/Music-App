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
- Not yet tested: Windows `setup.bat`, CUDA processing, YouTube download (YouTube blocks cloud IPs).
- Owner's test song: https://www.youtube.com/watch?v=Nkdw4PyPvyI. Run it first on their PC and review the
  chord chart and sheet music with them.

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
  must stay on the PC. Still to decide with the owner: Cloudflare Tunnel (no account; link changes on restart),
  Tailscale Funnel (free account, permanent https link), or ngrok (free account, one permanent domain).
  Then add a `share.bat` that starts the tunnel next to run.bat, and refuse to start it without a password.
