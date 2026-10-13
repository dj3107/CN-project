# 4K explainer video (split)

The 4K file (3840×2160, 388 MB) is bigger than GitHub's 100 MB per-file limit, so it is stored as four parts.
Join them back into one playable file:

**macOS / Linux**
```bash
cat dynamic_firewall_explainer_4k.mp4.part* > dynamic_firewall_explainer_4k.mp4
```

**Windows (Command Prompt)**
```
copy /b dynamic_firewall_explainer_4k.mp4.part0+dynamic_firewall_explainer_4k.mp4.part1+dynamic_firewall_explainer_4k.mp4.part2+dynamic_firewall_explainer_4k.mp4.part3 dynamic_firewall_explainer_4k.mp4
```

Check the result (optional):
```bash
shasum -a 256 dynamic_firewall_explainer_4k.mp4
# c04058cf021e388d689ab3c6d9814494db1f6b0d4ae740ac7fce44c3058532b6
```

Subtitles: `../dynamic_firewall_explainer.srt` (same timings).
