# SDXL UI

**Stable Diffusion XL with the classic A1111 / Forge toolkit, without Jupyter and without Colab.**

SDXL is the bigger, sharper successor to SD 1.5 — native 1024×1024 pictures, much better prompt following, and its own large ecosystem of checkpoints and LoRAs. This app wraps it in the same plain shape as Kandinsky UI, Sana UI, Z-Image UI and SD 1.5 UI (run it, a page opens in your browser, type a prompt, pictures appear), with the **full SD 1.5 UI feature set** carried over and adapted to SDXL underneath. Simple on the surface, everything underneath, clean looking.

**Status: code-verified build.** This package has passed its full stub-test suite (UI construction, sampler mapping, the SDXL two-encoder prompt-weighting and blending maths, keyframe parsing, end-to-end generate flows against a stub pipeline). It has **not yet had its first picture on a real PC** — that proof comes from a fresh-folder install, and this page will be updated when it lands.

## What it does

**Model controls (top of the page, A1111-style)**

- **Checkpoint picker** — the built-in `stabilityai/stable-diffusion-xl-base-1.0`, or any single-file `.safetensors` / `.ckpt` dropped into the `models/` folder. Refresh button after adding one.
- **Clip skip** 1 or 2. 1 is the SDXL standard conditioning. 2 stops one CLIP layer earlier in both text encoders — compel has no mode for that, so it is encoded the way Diffusers itself does it, and **prompt weighting has no effect at clip skip 2** (the app says so in the status line and settings.txt whenever weighting is not active).
- **Two LoRA slots** — `.safetensors` files from the `loras/` folder, each with a weight slider (0–1.5). SDXL LoRAs only; SD 1.5 / SD 3.5 LoRAs do not fit this model.
- **Embeddings (textual inversion)** — `.pt` / `.safetensors` / `.bin` files from the `embeddings/` folder, ticked in the Embeddings accordion. See the Embeddings section below.
- **CPU offload** checkbox — off by default. SDXL in float16 is about 7 GB of weights, so a 12 GB card holds it entirely; offload is there for tighter setups or out-of-memory corners.
- No token, no license step: the base model is ungated (OpenRAIL++ license).

## Embeddings

Textual inversion embeddings work the A1111 way:

1. Drop the files (`.pt`, `.safetensors` or `.bin`) into the `embeddings/` folder and hit **Refresh list** in the Embeddings accordion.
2. Tick the embeddings you want loaded, then **type the embedding's name — the file name without its extension — in the prompt or the negative prompt** to use it. Most SDXL embeddings are negative-prompt ones (type the name in the negative prompt). The status line after a run names the embeddings that are loaded, and settings.txt logs them for every run.

Format notes:

- SDXL embeddings in the A1111 format carry vectors for **both** text encoders (the `clip_l` and `clip_g` entries). Multi-vector embeddings work: typing the name expands to all its vectors, in both encoders.
- A file with vectors for **only one encoder** loads for that encoder, and the status line says so.
- A file that doesn't load — wrong format, an SD 1.5 embedding, a name that collides with a built-in tokenizer word — is **reported in the status line, not fatal**: the run continues without it.
- Embeddings apply to every tab (they live on the shared model), re-apply cleanly when the checkpoint changes, and are gone after **Unload models**.

**Text to image**

- Prompt + negative prompt (default negative: `low quality, bad quality`), size presets (Square 1024×1024 / Portrait 832×1216 / Landscape 1216×832 / Wide 1344×768 / Tall 768×1344) plus width/height sliders up to 2048.
- Steps, CFG scale, seed, batch name, 1–10 images per Generate.
- **Samplers**: Euler a, Euler, DPM++ 2M Karras, DPM++ SDE Karras, DDIM, LMS — swapped per generation from the checkpoint's own scheduler config.
- **Prompt weighting**, A1111 syntax: `(word)`, `(word:1.2)`, `[word]`, via the compel library — in every tab. SDXL conditions on **two** text encoders; compel encodes each one the SDXL way (penultimate hidden states), the two are concatenated, and the pooled embedding comes from the second encoder. If compel ever fails to load, the app falls back to plain tokenization and *says so* in the status line and settings.txt.
- **Prompt blending** — a second prompt mixed into the first at the embedding level, weight 0–1. Both the sequence embeddings and the pooled embedding are mixed with the same weight.
- **Hires fix** — generate, Lanczos-upscale 1.5× or 2×, repaint at a second-pass denoise (default 0.35). Big second passes automatically use attention/VAE slicing so they fit the card.
- **ControlNet** — Canny (`diffusers/controlnet-canny-sdxl-1.0`), Depth (`diffusers/controlnet-depth-sdxl-1.0`) or OpenPose (`thibaud/controlnet-openpose-sdxl-1.0` — the diffusers org publishes no SDXL OpenPose ControlNet, so this uses Thibaud's, the standard one), with a control weight. Canny edges are extracted from your image automatically. **Depth and OpenPose take an already-preprocessed control image** (a depth map / pose skeleton) — there is no automatic depth or pose estimator in this version. Models download (~2.5 GB each) on first use. The hires second pass runs without ControlNet.
- Prompt queue (one prompt per line — the queue wins over the prompt box, all lines share one batch folder), `{a|b|c}` dynamic prompts re-rolled per image, Random prompt button (lines from `random_prompts.txt` next to the app, read fresh per click; edit the file to make it yours).

**Image to image** — upload a picture, repaint it at a denoise strength (default 0.6). Prompt weighting, samplers, LoRAs and clip skip all apply. The input image is saved into the batch folder as `input_000001.png` alongside the results.

**Inpaint** — upload an image and a mask (white = repaint); repaint runs at full denoise, as inpainting should. Image and mask are both saved into the batch folder.

**House features, as in the other apps**

- Batch folders under `images_out/<batch name>/`, reused across runs, frames numbered continuously (`000001.png`, `000002.png`, …).
- A `settings.txt` block per run logging **everything**: checkpoint, sampler, clip skip, whether prompt weighting was active, LoRA files and weights, active embeddings, blending / hires / ControlNet state, sizes, steps, CFG, and every seed — so any picture can be reproduced.
- Stop button on every tab, Star batch (copies the last batch into `images_out/favorites/`, never overwrites), Unload models, and a session history (last 10) — picking an entry restores its prompt *and* its settings.
- Memory is freed between images inside a run, so long queues don't creep upward.
- The VAE is held in float32 — the stock SDXL VAE is unreliable in float16 and can produce black or blotchy pictures. Everything else runs in float16.
- The safety checker is not loaded — you're responsible for what you generate, same as A1111.

## Animation tab

A port of **Animation Script v6.0 by Animator-Anon** (Deforum-inspired) — the longtime A1111 animation tool — re-implemented on this app's Diffusers machinery, since the original drives A1111 internals and can't run here literally. Identical in shape to the SD 1.5 UI port.

- **Positive / Negative Prompts template boxes** and a **Keyframes** box with the script's own line format: `time_s | prompt | positive | negative`, `template`, `prompt_from_png`, `transform` (zoom / shifts / rotation, per second), `seed`, `denoise`, `noise`, and `model` (switch checkpoint mid-animation by file name).
- Prompt keyframes **crossfade** into each other exactly like the script: full weight at a keyframe, ramping to the next prompt by the frame before its keyframe. The script did this with A1111's `AND` prompt composition; here the identical maths runs at the embedding level — sequence embeddings and pooled embeddings both crossfade. Prompt weighting works inside keyframe prompts (at clip skip 1).
- **Total Animation Length** (default 10 s) × **Framerate** (default 15) = frame count. Frame 1 is text-to-image; every later frame is the previous frame transformed (the script's `zoom_at2`: scale, rotate, paste onto a noise-and-blur background), optionally noised, then img2img at the frame's **Denoising Strength** (default 0.40).
- **Seed_March** (seed +1 per frame), **Smoothing_Frames** (the script's `morph` crossfades inserted between frames, 0–32), **Add_Noise** with **Noise Strength**.
- Frames land in the batch folder with the usual continuous numbering. settings.txt logs all animation settings plus **each frame's seed and prompt**, and an `animation_<timestamp>_frames.csv` in the batch folder records the full per-frame table.
- **Video**: GIF / MP4 / WEBM checkboxes (WEBM on by default, like the script) assemble the frames into a finished movie — **no system ffmpeg install needed**: the app uses the FFmpeg binary supplied by the imageio-ffmpeg package (a `tools/ffmpeg.exe` dropped next to the app, if one is ever present, takes precedence, and a system ffmpeg on PATH is the last resort). The movie **plays right in the page**, in the **Movie** player under the results: MP4 when one was made, else WEBM. A GIF on its own stays a file in the batch folder — GIFs do not play in the Movie player. Only if no ffmpeg can be found at all (now a rare corner) do the frames simply stay in the folder, with the status and settings.txt saying so plainly.
- **Credit**: video assembly uses FFmpeg (https://ffmpeg.org) provided by the imageio-ffmpeg package.
- Stop cancels between frames; whatever frames are saved stay saved. There is **no resume button in this version** — re-running starts a new run in the same folder (numbering continues), it does not pick up mid-animation.
- Keyframed values interpolate **linearly** between keyframes in this port (the script used a polynomial pass for some columns).
- Seed keyframes switch the seed outright, combined with the Seed_March rule (the script's A1111 subseed noise-blend between keyframed seeds is not ported).
- **Not ported**: the paste-props overlay system (`prop`, `set_stamp`, `clear_stamp`, `Poper_Folder`) and the `rendertext` text overlays (`set_text`, `clear_text`). External `source` video/image frame feeds and the `col_set` / `col_clear` colour-correction keyframes are also not applied. All of these are recognised in the Keyframes box, counted, and named in settings.txt — never silently dropped. The script's make-*.bat helper files are not written either.
- Note: animation frames are full SDXL generations, so an animation takes noticeably longer per frame than in SD 1.5 UI. Shorter lengths and smaller sizes are the practical levers.

## Honest exclusions in this version

- **No refiner pass** — SDXL's separate refiner model is a possible future extra; this version is feature parity with SD 1.5 UI, and the base model alone is the proven shape.
- **No VAE picker** — most SDXL checkpoints bake their VAE in, and the surface stays clean without it.
- **No face restoration** (GFPGAN / CodeFormer) — a possible future extra.
- No X/Y/Z plots, no extension system — deliberately. This is the A1111 greatest-hits set, not the whole attic.

## Install

1. Extract the zip anywhere (a short path like `C:\AI\sdxl-ui` is best).
2. Double-click **install.bat**. Python is bundled in `python_embeded\` — nothing to install first, no venv, no system Python touched. It installs PyTorch with CUDA support (the big download, ~2.5 GB), then the remaining packages.
3. Double-click **run.bat**. The app opens in your browser at a local address.

First Generate downloads the base model (~7 GB) into the Hugging Face cache on your user profile. A single-file checkpoint's first load also fetches the small SDXL config files once. ControlNet models download on first use.

## Folders

- `models/` — drop single-file checkpoints here (`.safetensors` / `.ckpt`), then hit **Refresh list** in the app.
- `loras/` — drop SDXL LoRAs here (`.safetensors`), then **Refresh list**.
- `embeddings/` — drop SDXL textual inversion embeddings here (`.pt` / `.safetensors` / `.bin`), then **Refresh list**.
- `images_out/` — your batches, one folder per batch name, `settings.txt` beside the frames; `images_out/favorites/` holds starred batches.
- `random_prompts.txt` — the Random prompt button's list; edit freely.

## Requirements

- Windows, an NVIDIA GPU (built and tested for an RTX 4070, 12 GB), a recent driver, and an internet connection for the installs and model downloads.
- No Python installation needed — it's bundled.
- SDXL is a bigger model than SD 1.5: at 1024×1024 expect each picture to take noticeably longer than SD 1.5 UI does. That's the model, not the app.

MIT licensed, like the other UIs in this family.
