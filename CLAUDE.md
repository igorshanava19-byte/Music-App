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

## Ideas for later (the owner liked these)
Pitch/key shifting of the audio, synced lyrics with chords above them (Whisper), chord diagrams,
a better piano transcription model, separate acoustic vs electric guitar.
