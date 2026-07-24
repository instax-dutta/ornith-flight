# Hyperframes Composition Brief: Ornith-Flight

## Objective
Create a short hype launch-style brag video for Ornith-flight — an open-source inference engine that runs 35B MoE models on consumer hardware.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: vertical — 1080x1920
- Duration: 20 seconds

## Source Material
- Project root: /Users/saiduttaabhishekdash/ornith-flight
- Primary files read: README.md, OPTIMIZATION_SUMMARY.md, QUICK_REFERENCE.md, PUBLICATION_READY.md
- Product name: Ornith-flight
- Tagline / strongest claim: "Run 35B parameter MoE models on consumer hardware through intelligent expert streaming"
- Key visual moment to recreate: The 4-tier memory hierarchy (T0 Resident -> T1 Hot-Store -> T2 LRU Cache -> T3 Disk Stream)
- Copy that must appear verbatim:
  - "35 BILLION PARAMETERS"
  - "ON 8GB OF RAM"
  - "66 of 7,168 experts in RAM"
  - "Before: Cannot run / Now: Runs comfortably"
  - "Large models. Any hardware."

## Creative Direction
- Tone preset: app-store
- Creative direction: hype tech product launch
- Interpretation: Feature-card clean with genuine excitement. Fast reveals. Big numbers. Real benchmarks as proof. Energetic but not chaotic.
- Angle: The video bridges the gap between "35B parameters" and "8GB laptop" — the absurd claim is real, backed by actual benchmarks. This is a genuine product launch, not a parody.
- Hook: "35 BILLION PARAMETERS" slams in. Beat. "ON 8GB OF RAM." slides beneath.
- Outro / punchline: "ORNITH-FLIGHT" centered. Tagline: "Large models. Any hardware."
- Avoid:
  - Generic SaaS language ("streamline your workflow")
  - Abstract filler visuals
  - Emojis in text

## Visual Identity
- Background: #0a0a0f (deep dark)
- Text: #ffffff (white)
- Accent: #3b82f6 (electric blue)
- Secondary accent: #8b5cf6 (purple)
- Success green: #22c55e
- Error red: #ef4444
- Display font: Inter, bold heavy weights (fallback: system sans-serif)
- Body font: Inter, medium weight (fallback: system sans-serif)
- Visual references from the project:
  - 4-tier memory pyramid from README
  - Comparison table: Before/After
  - Benchmark numbers from optimization summary

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. The Hook — 2.5s — "35 BILLION PARAMETERS" then "ON 8GB OF RAM"
2. The Gap — 3s — "REQUIRED: 70GB+ VRAM" vs "ACTUAL: 5.6GB" with subtitle
3. The Architecture — 5.5s — 4-tier memory pyramid cascading T0->T1->T2->T3
4. The Results — 5s — Two benchmark stat cards + Before/Now comparison
5. Outro — 4s — "ORNITH-FLIGHT" logo with tagline

## Audio
- Audio role: energetic tech bed
- Audio arc: builds energy through scenes 1-3, peaks at scene 4 results, resolves at scene 5 logo
- Music: happy-beats-business-moves-vol-1-by-ende-dot-app.mp3
- Music treatment: volume ~0.38, slight fade-in over 0.5s, small dip at 16s for outro logo, hold through end
- Music cue guidance: bundled preset at assets/music/cues/happy-beats-business-moves-vol-1-by-ende-dot-app.music-cues.json (120 BPM). Strong cues at 17.02s and 18.02s — aim to land the Outro logo reveal near 17-18s. Beat grid at ~0.5s intervals for sequential stat card reveals in Scene 4.
- Audio-reactive treatment: subtle — use RMS/bass for hero text glow and card presence breathing. No waveform/equalizer.
- Audio-coupled moments:
  - Scene 1 hook text — impact accent on "35 BILLION PARAMETERS" slam-in
  - Scene 3 tiers — soft card/drop sound per tier cascade
  - Scene 4 stat cards — card-slide sounds for each card entry
  - Scene 5 logo — impactBell_heavy accent on logo settle
- SFX selection guidance: clean interface sounds for card/stat reveals (interface/drop, interface/click). Impact sounds for major moments (impactSoft_medium for hook, impactBell_heavy for outro). No glitch/error/chaotic sounds.
- SFX analysis guidance: prefer low/medium HF risk for polished feel
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: copy the chosen music into `brag-output/composition/assets/music/`. Hyperframes will handle SFX.

## Hyperframes Instructions
Use the current `hyperframes` skill and CLI workflow. Prefer native Hyperframes conventions.

Requirements:
- 20 seconds total, vertical 1080x1920
- Show the 4-tier memory hierarchy as an animated cascade
- Show at least one stat or metric from the actual project benchmarks
- Keep all text readable — no text faster than 0.3s per word settled
- Include music and clean SFX layer
- Use the bundled music cue preset for timing guidance. Lock at least 1 major reveal to a strong cue (±0.15s). Snap sequential tier reveals to consecutive beats (±0.10s).
- Subtle audio-reactive treatment on hero text glow or background
- Honor the exact color palette
- Run hyperframes lint and validate before render