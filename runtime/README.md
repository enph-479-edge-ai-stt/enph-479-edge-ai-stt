# runtime/ — on-board program (ARM under PYNQ Linux)

The live demo program on the KV260. Chain:

    USB mic -> ALSA/sounddevice -> ring buffer -> numpy features (bit-matched)
    -> PYNQ DMA -> PL (AM + char-LM) -> C decoder + kenlm -> text
    -> websocket -> browser dashboard

## Planned modules

- `capture/`     ALSA + sounddevice ring buffer
- `features_np/` numpy reimplementation of the training feature pipeline; **must bit-match `../shared/golden/`**
- `pynq_driver/` PYNQ Overlay / allocate / DMA / MMIO wrapper
- `decoder/`     beam search + kenlm; Python prototype, then C port (the one real-time-critical component)
- `frontend/`    websocket server + browser dashboard (waveform / probs / text / latency / power)
- `bare_metal/`  stretch: same chain in C with no OS, for the low-power demo

(Folders get created as each module starts; documented here first.)
