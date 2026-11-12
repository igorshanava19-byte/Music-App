# Music App

A personal tool that takes a song (a YouTube link or an audio file) and gives you:

- **A minus track** (the song with the vocals removed), plus a vocals-only track
- **Every instrument on its own** (vocals, guitar, piano/keys, bass, drums, other) in a stem
  mixer where you can mute or solo parts, slow the song down without changing the pitch, and loop sections
- **A chord chart** bar by bar, with key and tempo, transpose, Nashville numbers and a capo tip for guitar
- **Sheet music for each instrument it hears**: guitar and bass as notation + tab with chord names,
  piano/keys on a grand staff, the vocal melody, and drums. You can play it back in the browser, print it to PDF,
  or download MusicXML (opens in MuseScore / Guitar Pro for editing) and MIDI

Everything runs locally on your computer. An NVIDIA GPU makes it much faster
(about 30 seconds per song instead of a few minutes).

## Install (Windows)

1. Download this repository into a folder, e.g. `E:\Documents\Music App`
   (green **Code** button → **Download ZIP**, then unzip, or `git clone`).
2. Double-click **`setup.bat`**. It installs everything into a `.venv` folder inside the project
   (about 4 GB, mostly PyTorch and its CUDA libraries) and downloads the AI models. It takes 5–15 minutes.
3. Double-click **`run.bat`**. Your browser opens at <http://127.0.0.1:8000>.

On macOS/Linux use `./setup.sh` and `./run.sh` instead.

## Using it

Paste a YouTube link (or drop an audio file) and click **Process**. Processing runs in the background,
one song at a time, and the songs stay in your library (stored in the `data` folder).

If YouTube says *"Sign in to confirm you're not a bot"*, open `settings.bat`, turn on the
`MUSICAPP_COOKIES_FROM_BROWSER` line with the browser you're signed in to YouTube with, and restart `run.bat`.

**Open it from anywhere (phone, another computer):** set a password of 12+ characters on the
`MUSICAPP_PASSWORD` line in `settings.bat`, close `run.bat`, and double-click **`share.bat`** instead. It prints a
`https://….trycloudflare.com` link (a free Cloudflare quick tunnel to your PC; no account needed). The link changes
every time you start `share.bat`, and your PC must stay on while you use it.

> Downloading from YouTube is against YouTube's terms of service and the songs are copyrighted.
> This is meant as a personal practice tool. Don't redistribute what it produces.

## How it works

| Step | Tool |
| --- | --- |
| Download | [yt-dlp](https://github.com/yt-dlp/yt-dlp) (+ Deno for YouTube's JavaScript) |
| Split into 6 stems | [Demucs](https://github.com/facebookresearch/demucs) `htdemucs_6s` |
| Minus track | song minus the vocal estimate (`htdemucs_ft` on GPU for the cleanest vocals) |
| Tempo, beats, key | librosa beat tracking + Krumhansl key profiles |
| Chords | chroma templates + bass-note root detection, smoothed with a Viterbi decoder |
| Notes | Spotify [Basic Pitch](https://github.com/spotify/basic-pitch) per stem (ONNX), band-split onset detection for drums |
| Sheet music | notes quantized to the beat grid, guitar/bass fingering search for tabs, MusicXML writer |
| Viewer | [alphaTab](https://alphatab.net) (bundled in `web/vendor`) |

**What to expect:** the minus track and stems are very good. Chords are usually right but not always.
Sheet music is a *draft*: bass lines and melodies come out well, while dense piano and guitar parts will have
wrong or extra notes. Open the MusicXML in [MuseScore](https://musescore.org) to clean it up. Demucs puts all
guitars (acoustic and electric) into one guitar stem, and anything that isn't vocals/drums/bass/guitar/piano
(strings, synths, horns…) ends up in "Other".

## Project layout

```
app/            Python backend (FastAPI)
  main.py         web server + API
  jobs.py         background job queue
  pipeline.py     the processing steps for one song
  separation.py   Demucs stems + minus track
  analysis.py     tempo/beats, key, chord recognition
  transcription.py  audio -> notes (Basic Pitch, drums)
  notation.py     notes -> MusicXML/MIDI (quantizing, tab fingering)
  check.py        setup self-check (python -m app.check)
web/            the browser UI (plain HTML/CSS/JS, no build step)
tools/          make_test_song.py: synthetic song with a known answer key
```

Settings can also be given as environment variables: `MUSICAPP_PORT`, `MUSICAPP_DATA_DIR`,
`MUSICAPP_DEVICE` (`cuda`/`cpu`), `MUSICAPP_STEMS_MODEL`, `MUSICAPP_MINUS_MODEL`, `MUSICAPP_COOKIES_FROM_BROWSER`.
