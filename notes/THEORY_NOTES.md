# Theory Notes — Instrument Recognition Project

**What this file is:** Running revision notes from a theory walkthrough on audio/DSP fundamentals,
building up to the ML pipeline for a musical instrument recognition project. Written module by
module, in the order they were taught, so it can be read top-to-bottom as a self-contained primer.
No code here — pure theory/intuition. See the project's other docs for implementation.

**Author context:** Student has no prior background in audio/signal processing. Notes are written
at that level — intuition first, math only where it earns its keep.

**Status:** All 7 modules complete. Theory arc done — next step is building the project itself
(dataset download, preprocessing code, model training).

---

## Module 1: What Sound Physically Is

- Sound is a **pressure wave**: a vibrating source (e.g. a guitar string) pushes air molecules,
  creating alternating regions of higher/lower air pressure that travel outward. A microphone or
  eardrum measures pressure at one point over time.
- So a sound, fundamentally, is **one number (pressure deviation) changing over time**. Plotted,
  this is a **waveform**.
- Two key properties of a waveform:
  - **Frequency** — how many oscillations per second, in **Hertz (Hz)**. Perceived as **pitch**.
    - Low frequency → low note (e.g. bass ~80 Hz). High frequency → high note.
    - Human hearing range: roughly **20 Hz – 20,000 Hz**.
  - **Amplitude** — size of the pressure swing. Perceived as **loudness**.
- **Important seed idea for later:** most real-world sounds (a single instrument note included)
  are *not* one clean frequency — they're many frequencies layered together simultaneously.
  This fact is the basis for:
  - the **Fourier Transform** (Module 3) — decomposing a sound into its constituent frequencies.
  - **timbre** (Module 6) — why a violin and flute playing the "same note" (same fundamental
    frequency) still sound completely different.

**One-line takeaway:** *Sound = pressure over time → frequency gives pitch, amplitude gives
loudness, and real sounds are mixtures of many frequencies at once.*

---

## Module 2: Digitizing Sound

- A microphone captures a **continuous** pressure signal; a computer can only store a finite list
  of numbers, so digitizing = deciding (1) how often to measure, and (2) how precisely.

**1. Sampling Rate — "how often do we measure?"**
- We measure the pressure value at regular intervals; each measurement = one **sample**.
- **Sample rate** = samples per second, in Hz. E.g. a 3s clip at 44,100 Hz = an array of
  44,100 × 3 = 132,300 numbers. That array *is* the digital audio file.
- Common rates: **44,100 Hz** (CD/music standard), **22,050 Hz** or **16,000 Hz** (common in ML
  pipelines — cheaper to process).

**Nyquist theorem (key idea of this module):**
- To faithfully capture a frequency of `F` Hz, you must sample at **≥ 2×F Hz**. Sample too slowly
  and you get **aliasing** — the true frequency is misread as a false, lower one.
- 44,100 Hz sampling → captures frequencies up to 22,050 Hz (its "Nyquist frequency"), comfortably
  covering human hearing (~20,000 Hz max).
- Downsampling to 16,000 Hz in an ML pipeline caps captured frequency at 8,000 Hz — a deliberate
  trade-off (usually fine, since most instrument/voice energy sits below 8 kHz).

**2. Bit Depth — "how precisely do we measure each sample?"**
- Each sample's value is rounded to the nearest representable number — this rounding is
  **quantization**.
- **16-bit audio** (CD standard): 2¹⁶ = 65,536 possible values per sample. Higher bit depth = less
  quantization error, more storage. 16-bit is plenty for this project.
- Sampling rate = "horizontal" (time) resolution; bit depth = "vertical" (amplitude) resolution.

**One-line takeaway:** *A digital audio file is a waveform sliced into discrete time-steps (sample
rate) with each value rounded to fixed precision (bit depth); Nyquist's theorem requires sample
rate ≥ 2× the highest frequency you want to capture faithfully.*

---

## Sidebar: Handling Variable-Length Audio Input (no "EOS tokens" in audio)

*(Asked ahead of schedule — properly belongs with Module 7's preprocessing pipeline, kept here too
since it came up in this order.)*

- Text sequence models can use a special **EOS token** to mark "stop here" because text is
  **discrete symbols** — adding one more symbol is cheap and unambiguous.
- Raw audio/spectrograms are **continuous-valued** (Module 1/2) — there's no discrete vocabulary
  to add a special "stop" symbol to, so the EOS-token trick doesn't apply to audio
  *classification* tasks. (Some audio *generation* models like Whisper do tokenize audio and use
  EOS-style tokens — but that's for sequence generation, not classification, and not relevant
  here.)
- What's used instead, typically both together:
  1. **Fixed-length windowing** (the primary technique for this project) — chop the whole piece
     into fixed-length chunks (e.g. every 3s, per the IRMAS convention), classify each window
     independently, then aggregate predictions (majority vote / averaged probabilities) across
     windows for a whole-song answer. Sidesteps the length problem for most of the data.
  2. **Silence-padding** — for a short clip or a leftover partial window (e.g. a song's length not
     evenly divisible by the window size), pad with literal **zeros** (= silence, physically
     meaningful "nothing here," per Module 1) rather than an arbitrary token.
  3. **Masking** (optional) — a parallel 1/0 array marking real-audio vs. padded positions, so the
     model doesn't treat padding as signal. More relevant for RNNs/Transformers than plain CNNs.

**One-line takeaway:** *No EOS tokens in audio classification — normalize length upfront via
fixed-length windowing, and silence-pad (zeros) any leftover short/partial clips.*

---

## Module 3: Time Domain → Frequency Domain (Fourier Transform)

- **Core claim:** any complex sound wave can be decomposed into a sum of simple sine waves, each
  with its own frequency, amplitude, and phase. (True for any reasonably well-behaved signal —
  real audio always qualifies.)
- **Chord analogy:** three piano notes played together arrive at the mic as one tangled waveform —
  you can't visually pick out the individual notes. The **Fourier Transform (FT)** takes that
  tangled signal and returns the "recipe": which frequencies are present and how strongly.
- **Prism analogy:** white light looks like one thing but is really many wavelengths mixed
  together; a prism separates them. The FT is a mathematical prism for sound.
- **Two views of the same sound:**
  - **Time domain** (Modules 1–2): amplitude vs. time — "what does pressure do moment to moment?"
  - **Frequency domain** (FT output, called a **spectrum**): amplitude vs. frequency — "how much
    of each frequency is present overall?"
- **DFT vs. FFT:** since digitized audio is a finite array of numbers (Module 2), we use the
  **Discrete Fourier Transform (DFT)** — the sampled-data version of this math. **FFT (Fast
  Fourier Transform)** is not a different concept — it's just a fast algorithm for computing the
  same DFT. Treat the two terms as interchangeable in conversation.
- **Crucial limitation (→ motivates Module 4):** taking the FT of an entire clip at once gives
  back *one static spectrum* for the whole thing — averaged across all time, with no sense of
  *when* each frequency occurred. A flute in the intro and a trumpet in the outro would blend into
  one meaningless snapshot. We need frequency **and** time together.

**One-line takeaway:** *The Fourier Transform decomposes sound into its constituent frequencies
(time domain → frequency domain), but applied to a whole clip at once it loses all timing
information — which is exactly the problem spectrograms (Module 4) solve.*

---

## Module 4: Spectrograms

- **Goal:** need frequency info *and* timing info together (Module 3's FT-on-whole-clip loses
  timing). Solution: **Short-Time Fourier Transform (STFT)**.
- **How STFT works:**
  1. Chop the waveform into many small overlapping windows (e.g. ~25ms each).
  2. Run the FFT on each window separately → a mini spectrum per window.
  3. Line the spectra up side by side in time order.
  - Result: a 2D grid — **x-axis = time, y-axis = frequency, brightness = energy at that
    time/frequency**. This 2D grid/array is the **spectrogram**. ("STFT" = the computation,
    "spectrogram" = the resulting picture/array — used near-interchangeably.)
  - Key insight: a spectrogram is literally a picture (time × frequency × intensity) — this is
    *why* instrument recognition gets treated as image classification with CNNs.
- **Time–frequency trade-off (uncertainty principle):**
  - Longer window → better frequency resolution, worse time resolution (blurs *when* things
    happen, e.g. smears a sharp drum hit).
  - Shorter window → better time resolution, worse frequency resolution (can't distinguish close
    frequencies).
  - Can't maximize both at once; typical middle ground ~20–40ms windows for music/speech.
- **Hop size:** windows are overlapped, sliding forward by a hop size smaller than the window
  itself (common: hop = 1/4 window length) → smoother/more detailed spectrogram, more compute.
  - *(Minor detail: each window is tapered with a window function, e.g. Hann window, before FFT,
    to avoid edge artifacts — standard/automatic in any audio library, no math needed.)*
- **Decibel (log) scaling:** raw FFT magnitudes span a huge range, and human loudness perception is
  roughly logarithmic — so spectrograms are typically converted to a **log/dB scale**
  ("log-magnitude spectrogram") before use, both for better visualization and closer alignment to
  human perception.
- **Remaining wrinkle (→ motivates Module 5):** the frequency axis here is linear in Hz, but human
  *pitch* perception isn't linear — motivates the Mel scale.

**One-line takeaway:** *A spectrogram is built by running the FT on many small overlapping
time-windows and stacking results side by side into a time-vs-frequency "image"; window size
trades time precision against frequency precision, and magnitudes are usually log-scaled (dB)
before use.*

---

## Module 5: Psychoacoustics & the Mel Scale

- **Problem from Module 4:** a normal spectrogram's frequency axis is linear in Hz, but human
  pitch perception is **not linear** — much more sensitive at low frequencies than high.
  E.g. 100→200 Hz sounds like a big pitch jump; 10,000→10,100 Hz (same 100 Hz gap) is barely
  noticeable. We perceive pitch closer to frequency *ratios* than raw Hz differences.
- **Mel scale:** empirically derived (1930s listening experiments, Stevens/Volkmann/Newman) — a
  scale where equal distances in "mels" sound like equal pitch differences to listeners. Roughly
  linear at low Hz, increasingly compressed (log-like) at high Hz.
  Approximation formula (not required to memorize): `mel = 2595 × log10(1 + f/700)`.
- **Mel filterbank (spectrogram → mel-spectrogram):** not just relabeling the axis — apply a set
  of overlapping **triangular filters** spaced along the mel scale (tightly packed at low
  frequencies, wide apart at high frequencies). Each filter sums energy from a range of the
  original linear-Hz bins into one **mel bin**.
  - Reduces e.g. 1000+ raw FFT bins down to ~40/64/128 mel bins.
  - Two wins: axis now matches human hearing perceptually, *and* the representation is far more
    compact (helps CNN training: less noise, fewer redundant bins).
  - Amplitudes are then log-scaled too → **log-mel spectrogram** — the actual standard input fed
    to CNNs in most modern audio classification papers, instrument recognition included.
- **MFCCs (Mel-Frequency Cepstral Coefficients)** — related, older technique:
  - Takes the log-mel spectrogram and applies a **DCT (Discrete Cosine Transform)**, keeping only
    the first ~13–40 coefficients per frame.
  - Historically dominant for speech/speaker recognition because it decorrelates/compresses
    features — needed by older models (e.g. GMMs) that can't learn feature correlations
    themselves.
  - **Not needed for our CNN-based approach** — a CNN can learn structure directly from the
    richer, less-compressed log-mel spectrogram. Good to recognize the term in papers, not what
    we'll build with.

**One-line takeaway:** *Human pitch perception is non-linear (more sensitive at low frequencies);
the mel scale captures this empirically, and a mel filterbank remaps a spectrogram's linear-Hz
axis into perceptually-spaced mel bins, producing the log-mel spectrogram — the actual standard
model input. MFCCs add a further DCT compression step useful for older ML methods, not needed for
CNNs.*

---

## Module 6: Timbre — What Makes Instruments Sound Different

- **The puzzle:** a violin and a flute playing the *same pitch* (same fundamental frequency) still
  sound clearly different. That quality is **timbre** ("tone color") — exactly what an instrument
  classifier must learn to detect.
- **1. Harmonics / overtones (main ingredient):** a vibrating string/air column vibrates
  simultaneously at the fundamental frequency *and* whole-number multiples of it (2×, 3×, 4×...).
  A "single note" is really a small stack of related frequencies (ties back to Module 3: complex
  waves = sums of sine waves).
  - What differs by instrument = the **relative loudness of each harmonic** (the **spectral
    envelope**) — an instrument's frequency "fingerprint."
  - Flute ≈ pure sine wave (mostly fundamental, weak harmonics). Clarinet emphasizes odd
    harmonics only (cylindrical closed tube physics) → hollow tone. Violin has a dense, rich
    harmonic stack → bright/complex tone.
  - This is directly what a spectrogram visualizes: harmonics = stack of horizontal bands at
    evenly-spaced frequencies; the brightness pattern across bands is the instrument's visual
    signature. **Main reason spectrogram+CNN classification works at all.**
- **2. Formants:** an instrument's physical body (wood cavity, tube bore, vocal tract) has its own
  fixed resonant frequencies that boost/dampen certain ranges **regardless of pitch played**.
  - Explains why an instrument sounds like itself across its whole pitch range — formants stay put
    while the harmonic stack shifts with the note.
  - On a spectrogram: a stable frequency "hump" that persists across notes/time.
- **3. ADSR envelope (time-domain signature):** how loudness evolves over a note's lifetime —
  **A**ttack (rise to peak), **D**ecay (initial drop), **S**ustain (held level), **R**elease
  (fade-out after note ends).
  - Piano/guitar (plucked/struck): near-instant attack, continuous decay, no real sustain plateau.
  - Violin/organ/winds (bowed/blown): slower attack, genuine flat sustain while played.
  - On a spectrogram: the vertical brightness-over-time pattern (sharp fading strike vs. long
    steady band).
- **4. Transients (bonus):** brief, noisy, non-tonal bursts at note onset (bow catching a string,
  a flute's breathy "chiff," a guitar pluck-click) — short-lived, broadband energy (not clean
  harmonic lines), an extra distinguishing cue.
- **Synthesis:** Timbre = spectral envelope (harmonics) + formants (body resonance) + ADSR
  (temporal shape) + onset transients. All four are physically real and all four leave visible,
  learnable traces on a log-mel spectrogram — the actual physical reason a CNN can learn to tell
  instruments apart from spectrogram "images."

**One-line takeaway:** *Timbre — what distinguishes instruments at the same pitch — comes from
harmonic/overtone strength (spectral envelope), fixed body resonances (formants), the note's
loudness-over-time shape (ADSR), and onset transients; all four are visible in a log-mel
spectrogram, which is why this classification task is solvable at all.*

---

## Module 7: Bridging to ML

- **Spectrogram → image:** a log-mel spectrogram (mel-bins × time-frames × log-magnitude) is
  literally a single-channel image → this is why the task is framed as CNN-based image
  classification. CNN filters learn to detect the Module 6 patterns (harmonic bands, formant
  humps, ADSR shapes).
  - Nuance: translation invariance (fine along the time axis) is shakier along the frequency
    axis (a shifted pattern = a different pitch, not a repositioned copy) — known wrinkle, plain
    2D CNNs still work well as a baseline regardless.
- **Task framing:**
  - **Single/predominant-label:** one label per clip → **softmax** + **cross-entropy loss** →
    prediction = argmax.
  - **Multi-label** (realistic — real songs have several instruments at once): independent
    **sigmoid** output per instrument (probability "present") + **binary cross-entropy (BCE)**
    per class → prediction = threshold each class independently (e.g. >0.5).
- **Full-song handling (closes the loop on the earlier windowing/padding sidebar):** slide a
  window (e.g. 3s) across the whole track, classify each window independently, then **aggregate**:
  - Single-label → majority vote across windows.
  - Multi-label → max or average predicted probability per instrument across all windows.
- **Model architecture — two realistic options:**
  - Small CNN trained from scratch (few conv+pool layers + dense head) — reasonable baseline on
    IRMAS-sized data.
  - **Transfer learning from pretrained audio embeddings** (PANNs, YAMNet, VGGish, OpenL3 —
    pretrained on large datasets like AudioSet) + a small classifier head — usually outperforms
    from-scratch training with limited labeled data; the modern practical default.
- **Evaluation metrics:**
  - Single-label: accuracy + per-class precision/recall/F1 + confusion matrix (datasets are
    usually class-imbalanced, so accuracy alone can hide a model ignoring rare classes).
  - Multi-label: plain accuracy doesn't apply well. Use per-class precision/recall/F1, aggregated
    as **macro-F1** (equal weight per class — surfaces rare-class neglect) and **micro-F1**
    (globally pooled — dominated by common classes). IRMAS/OpenMIC papers report both; good
    default for us too.
- **Full pipeline, start to finish:**
  `raw audio → resample (Mod 2) → chop into fixed windows, pad leftovers (sidebar) → STFT
  (Mod 4) → mel filterbank + log scale (Mod 5) → log-mel spectrogram per window → CNN (scratch or
  pretrained embeddings + head) → per-window prediction (softmax/sigmoid) → aggregate across
  windows → final instrument(s) for the whole song`

**One-line takeaway:** *A log-mel spectrogram is treated as a grayscale image for a CNN;
instrument recognition is single-label (softmax) or, more realistically, multi-label (per-class
sigmoid + BCE); full songs are handled via sliding-window prediction + aggregation; and evaluation
uses per-class F1 (macro/micro) rather than plain accuracy once multiple instruments can co-occur.*

---

## Summary: Full Theory Arc in One Diagram

```
Sound (pressure wave, Mod 1)
   → sampled at regular intervals → digital waveform (Mod 2, Nyquist = sample_rate/2)
   → chopped into overlapping windows (Mod 4)
   → FFT per window (Mod 3) → spectrum per window
   → stacked across time → spectrogram (time × linear-Hz × magnitude, Mod 4)
   → mel filterbank + log scale (Mod 5) → log-mel spectrogram (the "image")
   → [physically encodes timbre: harmonics/formants/ADSR/transients, Mod 6]
   → CNN (or pretrained embeddings + head) → per-window prediction (Mod 7)
   → aggregate across windows → predicted instrument(s) for the whole song
```

**Theory arc status: COMPLETE (Modules 1–7 + 1 sidebar).** Next: apply this to the actual project —
dataset acquisition (IRMAS to start), preprocessing code, and model implementation.
