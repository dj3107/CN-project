"""Explainer video for the Dynamic Firewall project (v2).

Narration: Kokoro TTS (one clip per sentence). Music + sound effects: synthesised with numpy.
Visuals: Pillow frames drawn in 1920x1080 design units through a camera that scales to the
output resolution. Frames are rendered in parallel segments and encoded with ffmpeg.

Usage:
  python3 make_video2.py OUT.mp4 [--height 1440] [--workers 4]
  python3 make_video2.py OUT.mp4 --preview SCENE      # 4 stills of one scene
  python3 make_video2.py OUT.mp4 --bench              # time a few frames
"""
import argparse, hashlib, math, os, random, subprocess, sys, time
import multiprocessing as mp
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DW, DH, FPS = 1920, 1080, 30           # design units
VOICE_SR, SR = 24000, 48000
LEAD, GAP, TAIL = 0.7, 0.4, 1.0
VOICE, SPEED = "af_heart", 1.05

# ---------------------------------------------------------------- palette
BG = (10, 14, 28)
PANEL = (22, 30, 52)
GRID = (52, 66, 98)
FG = (236, 241, 250)
MUTED = (148, 162, 190)
BLUE = (88, 160, 255)
GREEN = (72, 214, 140)
RED = (255, 92, 104)
ORANGE = (255, 170, 70)
PURPLE = (176, 132, 255)
TEAL = (64, 214, 214)
YELLOW = (255, 214, 90)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
FONT_FILES = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf",
              "mono": "DejaVuSansMono.ttf", "monob": "DejaVuSansMono-Bold.ttf"}
_fonts = {}


def font(size, kind="sans"):
    """Font at a *pixel* size (used for measuring in design units and, scaled, for drawing)."""
    size = max(4, int(round(size)))
    key = (size, kind)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_DIR + FONT_FILES[kind], size)
    return _fonts[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def back_out(x, s=1.9):
    x = clamp(x)
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def P(t, start, dur=0.5):
    if dur <= 0:
        return 1.0 if t >= start else 0.0
    return ease((t - start) / dur)


def mix(c, a=1.0, base=None):
    """Colour with alpha (frames are drawn in RGBA blend mode). `base` kept for old call sites."""
    return (int(c[0]), int(c[1]), int(c[2]), int(255 * clamp(a)))


def lerpc(c1, c2, f):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * clamp(f)) for i in range(3))


def lerp(p, q, f):
    return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)


# ---------------------------------------------------------------- camera / drawing proxy
class Cam:
    """ImageDraw proxy: design coordinates -> output pixels, with zoom around a focus point."""

    def __init__(self, img, scale, zoom=1.0, focus=(DW / 2, DH / 2), shake=(0, 0)):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")
        self.s, self.z, self.f, self.sh = scale, zoom, focus, shake
        self.k = scale * zoom

    def p(self, x, y):
        return (((x - self.f[0]) * self.z + DW / 2 + self.sh[0]) * self.s,
                ((y - self.f[1]) * self.z + DH / 2 + self.sh[1]) * self.s)

    def pts(self, pts):
        if pts and not isinstance(pts[0], (tuple, list)):
            pts = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
        return [self.p(*q) for q in pts]

    def rect(self, r):
        (x0, y0), (x1, y1) = self.p(r[0], r[1]), self.p(r[2], r[3])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def w(self, w):
        return max(1, int(round(w * self.k)))

    def line(self, pts, fill=None, width=1, joint=None):
        self.d.line(self.pts(pts), fill=fill, width=self.w(width), joint=joint)

    def rounded_rectangle(self, r, radius=0, fill=None, outline=None, width=1):
        rr = self.rect(r)
        rad = min(radius * self.k, (rr[2] - rr[0]) / 2 - 0.5, (rr[3] - rr[1]) / 2 - 0.5)
        if rr[2] - rr[0] < 2 or rr[3] - rr[1] < 2:
            return
        self.d.rounded_rectangle(rr, radius=max(0, rad), fill=fill, outline=outline,
                                 width=self.w(width) if outline else 0)

    def rectangle(self, r, fill=None, outline=None, width=1):
        self.d.rectangle(self.rect(r), fill=fill, outline=outline, width=self.w(width) if outline else 0)

    def ellipse(self, r, fill=None, outline=None, width=1):
        self.d.ellipse(self.rect(r), fill=fill, outline=outline, width=self.w(width) if outline else 0)

    def polygon(self, pts, fill=None, outline=None):
        self.d.polygon(self.pts(pts), fill=fill, outline=outline)

    def arc(self, r, start, end, fill=None, width=1):
        self.d.arc(self.rect(r), start, end, fill=fill, width=self.w(width))

    def text(self, xy, s, font=None, fill=None, anchor="la"):
        size = font.size * self.k
        kind = [k for k, v in FONT_FILES.items() if font.path.endswith(v)][0]
        self.d.text(self.p(*xy), s, font=globals()["font"](size, kind), fill=fill, anchor=anchor)


# ---------------------------------------------------------------- primitives (design units)
def text(d, xy, s, size=32, col=FG, a=1.0, kind="sans", anchor="mm", base=None):
    if a <= 0.01:
        return
    d.text(xy, s, font=font(size, kind), fill=mix(col, a), anchor=anchor)


def box(d, rect, col, a=1.0, fill=PANEL, r=18, w=3, base=None, glass=0.82):
    if a <= 0.01:
        return
    d.rounded_rectangle(rect, radius=r, fill=mix(fill, a * glass), outline=mix(col, a), width=w)


def line(d, pts, col, a=1.0, w=4, frac=1.0, base=None):
    if a <= 0.01 or frac <= 0:
        return
    d.line(path_prefix(pts, frac), fill=mix(col, a), width=w, joint="curve")


def path_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def path_point(pts, f):
    f = clamp(f)
    target = path_len(pts) * f
    for i in range(len(pts) - 1):
        seg = math.dist(pts[i], pts[i + 1])
        if target <= seg or i == len(pts) - 2:
            return lerp(pts[i], pts[i + 1], 0 if seg == 0 else clamp(target / seg))
        target -= seg
    return pts[-1]


def path_prefix(pts, f):
    if f >= 1:
        return pts
    total = path_len(pts) * f
    out = [pts[0]]
    for i in range(len(pts) - 1):
        seg = math.dist(pts[i], pts[i + 1])
        if total <= seg:
            out.append(lerp(pts[i], pts[i + 1], 0 if seg == 0 else total / seg))
            return out
        out.append(pts[i + 1])
        total -= seg
    return out


def dot(d, p, col, r=11, a=1.0, base=None):
    if a <= 0.01:
        return
    x, y = p
    d.ellipse((x - r * 2.2, y - r * 2.2, x + r * 2.2, y + r * 2.2), fill=mix(col, 0.10 * a))
    d.ellipse((x - r * 1.5, y - r * 1.5, x + r * 1.5, y + r * 1.5), fill=mix(col, 0.22 * a))
    d.ellipse((x - r, y - r, x + r, y + r), fill=mix(col, a))
    d.ellipse((x - r * 0.4, y - r * 0.4, x + r * 0.4, y + r * 0.4), fill=mix((255, 255, 255), 0.7 * a))


def ring(d, c, t, t0, col, dur=0.9, r0=30, r1=170, w=5):
    k = (t - t0) / dur
    if 0 <= k <= 1:
        r = r0 + (r1 - r0) * (1 - (1 - k) ** 3)
        d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), outline=mix(col, (1 - k) * 0.9), width=w)


def cross(d, p, col=RED, s=18, a=1.0, w=6):
    if a <= 0.01:
        return
    x, y = p
    d.line((x - s, y - s, x + s, y + s), fill=mix(col, a), width=w)
    d.line((x - s, y + s, x + s, y - s), fill=mix(col, a), width=w)


def arrowhead(d, p, q, col, a=1.0, s=16):
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    d.polygon([q, (q[0] - s * math.cos(ang - 0.45), q[1] - s * math.sin(ang - 0.45)),
               (q[0] - s * math.cos(ang + 0.45), q[1] - s * math.sin(ang + 0.45))], fill=mix(col, a))


def arrow(d, pts, col, a=1.0, w=5, frac=1.0):
    if a <= 0.01 or frac <= 0:
        return
    pp = path_prefix(pts, frac)
    d.line(pp, fill=mix(col, a), width=w, joint="curve")
    if len(pp) >= 2 and math.dist(pp[-2], pp[-1]) > 2:
        arrowhead(d, pp[-2], pp[-1], col, a)


def wrap_lines(s, size, maxw, kind="sans"):
    f = font(size, kind)
    lines, cur = [], ""
    for wd in s.split():
        trial = (cur + " " + wd).strip()
        if f.getlength(trial) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def para(d, xy, s, size, maxw, col=FG, a=1.0, kind="sans", spacing=1.35, base=None):
    for i, ln in enumerate(wrap_lines(s, size, maxw, kind)):
        text(d, (xy[0], xy[1] + i * size * spacing), ln, size, col, a, kind, anchor="la")


def title(d, t, s, sub=None, accent=TEAL):
    """Section title: letters slide up one after another."""
    f = font(46, "bold")
    x = 80
    for i, ch in enumerate(s):
        k = P(t, 0.05 + i * 0.018, 0.35)
        if k > 0.01:
            text(d, (x, 72 + (1 - k) * 26), ch, 46, FG, k, "bold", anchor="lm")
        x += f.getlength(ch)
    a = P(t, 0.3, 0.6)
    if sub:
        text(d, (82, 120), sub, 25, MUTED, a, "mono", anchor="lm")
    d.rounded_rectangle((80, 146, 80 + 300 * a, 152), radius=3, fill=mix(accent, a))


# ---------------------------------------------------------------- icons
def icon(d, kind, x, y, col, a, t=0.0, s=1.0):
    c = mix(col, a)
    if kind == "server":
        for i in range(3):
            yy = y - 20 * s + i * 14 * s
            d.rounded_rectangle((x - 20 * s, yy - 5 * s, x + 20 * s, yy + 7 * s), radius=3, outline=c, width=2)
            d.ellipse((x + 11 * s, yy - 1 * s, x + 15 * s, yy + 3 * s), fill=mix(GREEN, a))
    elif kind == "pc":
        d.rounded_rectangle((x - 21 * s, y - 18 * s, x + 21 * s, y + 9 * s), radius=4, outline=c, width=3)
        d.line((x, y + 9 * s, x, y + 17 * s), fill=c, width=3)
        d.line((x - 12 * s, y + 18 * s, x + 12 * s, y + 18 * s), fill=c, width=3)
    elif kind == "evil":
        icon(d, "pc", x, y, col, a, t, s)
        d.polygon([(x, y - 13 * s), (x - 9 * s, y + 4 * s), (x + 9 * s, y + 4 * s)], fill=c)
    elif kind == "switch":
        d.rounded_rectangle((x - 22 * s, y - 14 * s, x + 22 * s, y + 14 * s), radius=6, outline=c, width=3)
        arrow(d, [(x - 12 * s, y - 4 * s), (x + 13 * s, y - 4 * s)], col, a, 2)
        arrow(d, [(x + 12 * s, y + 5 * s), (x - 13 * s, y + 5 * s)], col, a, 2)
    elif kind == "gear":
        n, r1, r2 = 8, 22 * s, 16 * s
        pts = []
        for i in range(n * 4):
            ang = t * 0.9 + i * math.pi * 2 / (n * 4)
            r = r1 if (i % 4) in (1, 2) else r2
            pts.append((x + r * math.cos(ang), y + r * math.sin(ang)))
        d.polygon(pts, fill=c)
        d.ellipse((x - 7 * s, y - 7 * s, x + 7 * s, y + 7 * s), fill=mix(PANEL, a))
    elif kind == "globe":
        d.ellipse((x - 18 * s, y - 18 * s, x + 18 * s, y + 18 * s), outline=c, width=2)
        d.ellipse((x - 8 * s, y - 18 * s, x + 8 * s, y + 18 * s), outline=c, width=2)
        d.line((x - 18 * s, y, x + 18 * s, y), fill=c, width=2)


# ---------------------------------------------------------------- topology
H2, H3, S1, H1, CTL, NAT = (270, 380), (270, 800), (720, 590), (1150, 590), (720, 230), (1150, 230)
NODE_SIZE = {"host": (290, 120), "switch": (250, 112), "ctl": (310, 120), "nat": (230, 90)}
E_H2, E_H3, E_H1, E_OF, E_REST = [H2, S1], [H3, S1], [S1, H1], [S1, CTL], [H1, NAT, CTL]


def node(d, c, kind, label, sub, col, a=1.0, dim=False, ico="pc", t=0.0, glow=0.0):
    if a <= 0.01:
        return
    sc = back_out(a)
    w, h = NODE_SIZE[kind]
    w, h = w * sc, h * sc
    x, y = c
    edge = MUTED if dim else col
    if glow > 0.01:
        for k in range(3):
            g = 10 + k * 10
            d.rounded_rectangle((x - w / 2 - g, y - h / 2 - g, x + w / 2 + g, y + h / 2 + g),
                                radius=30 + g, fill=mix(col, 0.07 * glow * a))
    box(d, (x - w / 2, y - h / 2, x + w / 2, y + h / 2), edge, a,
        fill=(28, 30, 40) if dim else PANEL, r=50 if kind == "switch" else 18, w=4)
    ix = x - w / 2 + 44 * sc
    d.ellipse((ix - 30 * sc, y - 30 * sc, ix + 30 * sc, y + 30 * sc), fill=mix(edge, 0.13 * a))
    icon(d, ico, ix, y, edge, a, t if not dim else 0, sc)
    tx = x + 30 * sc
    text(d, (tx, y - 18 * sc), label, 30 * sc, FG if not dim else MUTED, a, "bold")
    text(d, (tx, y + 22 * sc), sub, 19 * sc, MUTED, a)


def link(d, pts, col, a, w, t, flow=True):
    """Link with a faint moving dash texture."""
    if a <= 0.01:
        return
    line(d, pts, col, a * 0.75, w)
    if flow:
        L = path_len(pts)
        n = int(L / 60)
        for i in range(n):
            f = ((i / n) + t * 0.08) % 1
            p = path_point(pts, f)
            d.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=mix(col, 0.5 * a))


def topo(d, t, show=None, ctl_dim=False, glow=None):
    s = show or {}
    g = lambda k: s.get(k, 1.0)
    gl = glow or {}
    link(d, E_H2, GRID, g("e_h2"), 6, t)
    link(d, E_H3, GRID, g("e_h3"), 6, t)
    link(d, E_H1, GRID, g("e_h1"), 6, t)
    link(d, E_OF, lerpc(BG, TEAL, 0.55), g("e_of"), 4, t)
    link(d, E_REST, lerpc(BG, PURPLE, 0.55), g("e_rest"), 4, t)
    node(d, H2, "host", "h2", "10.0.0.2 · trusted", GREEN, g("h2"), ico="pc", glow=gl.get("h2", 0))
    node(d, H3, "host", "h3", "10.0.0.3 · attacker", RED, g("h3"), ico="evil", glow=gl.get("h3", 0))
    node(d, H1, "host", "h1", "10.0.0.1 · log server", BLUE, g("h1"), ico="server", glow=gl.get("h1", 0))
    node(d, S1, "switch", "s1", "Open vSwitch", TEAL, g("s1"), ico="switch", glow=gl.get("s1", 0))
    node(d, CTL, "ctl", "Ryu", "SDN controller", PURPLE, g("ctl"), dim=ctl_dim, ico="gear", t=t, glow=gl.get("ctl", 0))
    node(d, NAT, "nat", "nat0", "10.0.0.254", MUTED, g("nat"), ico="globe")


def edge_label(d, p, s, col, a=1.0, size=20):
    if a <= 0.01:
        return
    w = font(size, "bold").getlength(s) + 22
    x, y = p
    d.rounded_rectangle((x - w / 2, y - 17, x + w / 2, y + 17), radius=11, fill=mix(BG, 0.92 * a), outline=mix(col, a), width=2)
    text(d, p, s, size, col, a, "bold")


def flow_packets(d, t, pts, times, travel, col, r=10, drop_after=None, drop_at=0.5, a=1.0, trail=True):
    for t0 in times:
        f = (t - t0) / travel
        if f < 0 or f > 1.4:
            continue
        dies = drop_after is not None and t0 + travel * drop_at >= drop_after
        end = drop_at if dies else 1.0
        if f <= end:
            if trail:
                for k in range(1, 5):
                    ff = f - k * 0.025
                    if ff > 0:
                        q = path_point(pts, ff)
                        rr = r * (1 - k * 0.15)
                        d.ellipse((q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr), fill=mix(col, a * (0.35 - k * 0.07)))
            dot(d, path_point(pts, f), col, r, a)
        elif dies and f <= drop_at + 0.4:
            k = (f - drop_at) / 0.4
            p = path_point(pts, drop_at)
            # shatter: fragments flying outwards
            for j in range(6):
                ang = j * 1.047 + t0 * 3
                q = (p[0] + math.cos(ang) * 40 * k, p[1] + math.sin(ang) * 40 * k)
                d.ellipse((q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4), fill=mix(RED, a * (1 - k)))
            cross(d, p, RED, 13, a * (1 - k), 5)


def pulse(d, t, pts, t0, travel, col, r=9, label=None):
    f = (t - t0) / travel
    if 0 <= f <= 1:
        for k in range(1, 6):
            ff = f - k * 0.03
            if ff > 0:
                q = path_point(pts, ff)
                rr = r * (1 - k * 0.12)
                d.ellipse((q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr), fill=mix(col, 0.3 - k * 0.05))
        p = path_point(pts, f)
        dot(d, p, col, r)
        if label:
            w = font(17, "monob").getlength(label) + 16
            d.rounded_rectangle((p[0] - w / 2, p[1] - 47, p[0] + w / 2, p[1] - 19), radius=8, fill=mix(BG, 0.85))
            text(d, (p[0], p[1] - 33), label, 17, col, 1, "monob")


def badge(d, xy, s, col, a=1.0, size=19, anchor="mm"):
    if a <= 0.01:
        return
    sc = back_out(a, 1.4)
    w = font(size, "mono").getlength(s) + 28
    x, y = xy
    if anchor == "lm":
        x += w / 2
    hw, hh = w / 2 * sc, 20 * sc
    d.rounded_rectangle((x - hw, y - hh, x + hw, y + hh), radius=10,
                        fill=mix((48, 18, 26) if col == RED else PANEL, 0.9 * a), outline=mix(col, a), width=3)
    text(d, (x, y), s, size * sc, col, a, "mono")


LOG_FILL = (12, 18, 32)


def log_panel(d, t, entries, rect=(1330, 180, 1880, 900), title_s="logs/security_events.log"):
    """Terminal panel; new lines type themselves in."""
    x0, y0, x1, y1 = rect
    d.rounded_rectangle(rect, radius=14, fill=mix(LOG_FILL, 0.9), outline=mix(GRID, 1), width=2)
    d.rounded_rectangle((x0, y0, x1, y0 + 50), radius=14, fill=mix((26, 34, 56), 0.95))
    for i, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse((x0 + 18 + i * 22, y0 + 19, x0 + 30 + i * 22, y0 + 31), fill=mix(col, 0.9))
    text(d, (x0 + 92, y0 + 26), title_s, 17, MUTED, 1, "mono", anchor="lm")
    lines = []
    for (te, s, col) in entries:
        if t >= te:
            n = int(len(s) * clamp((t - te) / max(0.25, len(s) / 160)))
            shown = s[:n]
            for k, ln in enumerate(wrap_lines(s, 18, x1 - x0 - 40, "mono")):
                # reveal per wrapped line
                start = sum(len(l) + 1 for l in wrap_lines(s, 18, x1 - x0 - 40, "mono")[:k])
                vis = ln[:max(0, n - start)]
                lines.append((("  " if k else "") + vis, col))
    lh = 29
    cap = int((y1 - y0 - 70) / lh)
    lines = lines[-cap:]
    for i, (ln, col) in enumerate(lines):
        text(d, (x0 + 20, y0 + 74 + i * lh), ln, 18, col, 1, "mono", anchor="lm")
    if (t * 2) % 2 < 1.2:
        yy = y0 + 74 + len(lines) * lh
        if yy < y1 - 10:
            d.rectangle((x0 + 20, yy - 10, x0 + 31, yy + 10), fill=mix(FG, 0.7))


# ================================================================ scenes
SCENES = []


def scene(sentences, tail=TAIL, accent=TEAL, energy=2, sfx=None, cam=None):
    def deco(fn):
        SCENES.append(dict(s=sentences, fn=fn, tail=tail, accent=accent, energy=energy,
                           sfx=sfx or (lambda c, D: []), cam=cam))
        return fn
    return deco


# --- 0 title ------------------------------------------------------------------
_rng = random.Random(7)
STARS = [(_rng.uniform(0, DW), _rng.uniform(0, DH), _rng.uniform(-14, 14), _rng.uniform(-10, 10)) for _ in range(46)]


@scene([
    "What if your network could spot an attack, and fight back on its own?",
    "This is the Dynamic Firewall with Security Event Logger: a software-defined networking project built with Mininet, the Ryu controller, and Python TCP sockets.",
    "Let's see what it does, and how it does it.",
], tail=1.2, accent=PURPLE, energy=1,
    sfx=lambda c, D: [(c[1] - 0.2, "boom")] + [(c[1] + 1.4 + i * 0.25, "pop") for i in range(4)])
def s_title(d, t, c, D):
    # constellation of nodes
    pts = []
    for (x, y, vx, vy) in STARS:
        pts.append(((x + vx * t) % DW, (y + vy * t) % DH))
    a0 = P(t, 0.1, 1.2)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dd = math.dist(pts[i], pts[j])
            if dd < 260:
                d.line((pts[i], pts[j]), fill=mix(TEAL, a0 * 0.22 * (1 - dd / 260)), width=2)
    for i, p in enumerate(pts):
        col = RED if i % 9 == 0 else (TEAL if i % 3 else PURPLE)
        dot(d, p, col, 4, a0 * 0.7)
    # occasional red "attack" packets getting stopped
    for k in range(6):
        t0 = 0.6 + k * 0.9
        f = (t - t0) / 1.2
        if 0 <= f <= 1 and t < c[1]:
            p0, p1 = pts[(k * 7) % len(pts)], pts[(k * 11 + 3) % len(pts)]
            dot(d, lerp(p0, p1, f), RED, 7)
    # question text first, then the title
    q = P(t, c[0], 0.6) * (1 - P(t, c[1] - 0.4, 0.4))
    text(d, (DW / 2, 520), "What if your network could fight back?", 64, FG, q, "bold")
    # title with letter reveal
    tt = t - c[1]
    if tt > -0.3:
        s1 = "Dynamic Firewall"
        f = font(104, "bold")
        x = DW / 2 - f.getlength(s1) / 2
        for i, ch in enumerate(s1):
            k = P(tt, i * 0.04, 0.45)
            if k > 0.01:
                text(d, (x, 380 + (1 - back_out(k)) * 60), ch, 104, FG, k, "bold", anchor="ls")
            x += f.getlength(ch)
        a = P(tt, 0.7, 0.6)
        text(d, (DW / 2, 450), "with Security Event Logger", 50, TEAL, a, "bold")
        d.rounded_rectangle((DW / 2 - 330 * a, 492, DW / 2 + 330 * a, 497), radius=3, fill=mix(TEAL, a))
        tags = [("Mininet", BLUE), ("Ryu SDN controller", PURPLE), ("OpenFlow 1.3", TEAL), ("Python TCP sockets", GREEN)]
        widths = [font(28, "bold").getlength(s) + 56 for s, _ in tags]
        x = DW / 2 - (sum(widths) + 30 * 3) / 2
        for i, (s, col) in enumerate(tags):
            b = P(tt, 1.4 + i * 0.25, 0.5)
            if b > 0.01:
                sc = back_out(b)
                cx = x + widths[i] / 2
                hw = widths[i] / 2 * sc
                d.rounded_rectangle((cx - hw, 580 - 32 * sc, cx + hw, 580 + 32 * sc), radius=32,
                                    fill=mix(PANEL, 0.85 * b), outline=mix(col, b), width=3)
                text(d, (cx, 580), s, 28 * sc, col, b, "bold")
            x += widths[i] + 30
        text(d, (DW / 2, 690), "SDN + TCP socket programming mini project · UE25CS243A, Project #1", 27, MUTED, P(tt, 2.6, 0.8))


# --- 1 idea -------------------------------------------------------------------
@scene([
    "A traditional firewall is static. Its rules are written in advance, and it can't react to what the application is actually seeing.",
    "This project closes that gap. A TCP log server watches its own clients, and when one misbehaves, it asks an SDN controller to change the switch's rules, while the network is running.",
    "And when the misbehaviour stops, the rules are removed again, automatically.",
], accent=TEAL, energy=1,
    sfx=lambda c, D: [(c[1] + 0.8 + i * 0.6, "pop") for i in range(3)] + [(c[2], "chime")])
def s_idea(d, t, c, D):
    title(d, t, "The idea", "let the application steer the network at run time")
    a = P(t, c[0], 0.6)
    ox = (1 - a) * -60
    box(d, (110 + ox, 220, 880 + ox, 880), MUTED, a)
    text(d, (495 + ox, 275), "Static firewall", 38, FG, a, "bold")
    icon(d, "server", 495 + ox, 345, MUTED, a, 0, 1.2)
    rules = ["allow tcp any -> 10.0.0.1:9000", "deny  tcp any -> 10.0.0.1:23", "allow icmp any -> any"]
    for i, r in enumerate(rules):
        b = P(t, c[0] + 0.5 + i * 0.3, 0.4)
        text(d, (170 + ox + (1 - b) * 30, 420 + i * 55), r, 26, MUTED, b * a, "mono", anchor="lm")
    text(d, (495 + ox, 600), "written once, by hand", 30, MUTED, a)
    if a > 0.5:
        d.line((170 + ox, 720, 820 + ox, 720), fill=mix(GRID, a), width=2)
        for k in range(7):
            f = ((t * 0.5 + k / 7) % 1)
            dot(d, (170 + ox + f * 650, 720), RED, 10, a)
        text(d, (495 + ox, 790), "a flood sails straight through", 27, RED, a)
    b = P(t, c[1], 0.6)
    ox = (1 - b) * 60
    box(d, (1040 + ox, 220, 1810 + ox, 880), TEAL, b)
    text(d, (1425 + ox, 275), "Dynamic firewall", 38, FG, b, "bold")
    pts = {"app": (1425 + ox, 420), "ctl": (1650 + ox, 670), "sw": (1200 + ox, 670)}
    labels = {"app": ("Log server", "detects", BLUE, "server"), "ctl": ("Controller", "decides", PURPLE, "gear"),
              "sw": ("Switch", "enforces", TEAL, "switch")}
    order = ["app", "ctl", "sw"]
    # orbiting packet around the loop
    lp = P(t, c[1] + 3.0, 0.6)
    if lp > 0:
        loop = [pts["app"], pts["ctl"], pts["sw"], pts["app"]]
        for k in range(3):
            dot(d, path_point(loop, (t * 0.25 + k / 3) % 1), YELLOW, 8, lp)
    for i in range(3):
        p, q = pts[order[i]], pts[order[(i + 1) % 3]]
        bb = P(t, c[1] + 2.5 + i * 0.5, 0.6)
        arrow(d, [lerp(p, q, 0.3), lerp(p, q, 0.7)], MUTED, bb, 4, bb)
    for i, k in enumerate(order):
        bb = P(t, c[1] + 0.8 + i * 0.6, 0.5)
        if bb <= 0.01:
            continue
        sc = back_out(bb)
        x, y = pts[k]
        box(d, (x - 125 * sc, y - 58 * sc, x + 125 * sc, y + 58 * sc), labels[k][2], bb)
        icon(d, labels[k][3], x - 80 * sc, y, labels[k][2], bb, t, sc * 0.9)
        text(d, (x + 22 * sc, y - 16 * sc), labels[k][0], 27 * sc, FG, bb, "bold")
        text(d, (x + 22 * sc, y + 20 * sc), labels[k][1], 22 * sc, labels[k][2], bb)
    ra = P(t, c[2], 0.6)
    text(d, (1425 + ox, 820), "rules expire on their own when traffic calms down", 25, GREEN, ra)


# --- 2 topology ---------------------------------------------------------------
@scene([
    "Here is the network, emulated in Mininet.",
    "Host h1 runs the log server, listening on two TCP ports: 9000 for logging, and 9001 for a protected admin service.",
    "Host h2 is a well-behaved, trusted client. Host h3 plays the attacker: it floods the server, or touches ports it shouldn't.",
    "All three hosts connect through a single Open vSwitch, s1, which speaks OpenFlow 1.3 to the Ryu controller.",
    "Because Mininet hosts live in isolated namespaces, a NAT node at 10.0.0.254 lets h1 reach the controller's REST API.",
], accent=BLUE, energy=2,
    sfx=lambda c, D: [(c[1], "pop"), (c[2], "pop"), (c[2] + 1.8, "pop"), (c[3], "pop"), (c[3] + 2.5, "pop"), (c[4], "pop")])
def s_topo(d, t, c, D):
    title(d, t, "The topology", "topology/firewall_topo.py")
    show = {"h1": P(t, c[1], 0.5), "h2": P(t, c[2], 0.5), "h3": P(t, c[2] + 1.8, 0.5),
            "s1": P(t, c[3], 0.5), "e_h1": P(t, c[3] + 0.3, 0.5), "e_h2": P(t, c[3] + 0.3, 0.5),
            "e_h3": P(t, c[3] + 0.3, 0.5), "ctl": P(t, c[3] + 2.5, 0.5), "e_of": P(t, c[3] + 3, 0.5),
            "nat": P(t, c[4], 0.5), "e_rest": P(t, c[4] + 1.5, 0.6)}
    # draw links progressively
    for k, pts in (("e_h1", E_H1), ("e_h2", E_H2), ("e_h3", E_H3), ("e_of", E_OF), ("e_rest", E_REST)):
        pass
    topo(d, t, show)
    for key, at in (("h1", c[1]), ("h2", c[2]), ("h3", c[2] + 1.8), ("s1", c[3]), ("ctl", c[3] + 2.5), ("nat", c[4])):
        pos = {"h1": H1, "h2": H2, "h3": H3, "s1": S1, "ctl": CTL, "nat": NAT}[key]
        ring(d, pos, t, at, {"h1": BLUE, "h2": GREEN, "h3": RED, "s1": TEAL, "ctl": PURPLE, "nat": MUTED}[key])
    badge(d, (H1[0], H1[1] + 105), ":9000 logging (rate-limited)", BLUE, P(t, c[1] + 1.5, 0.5), 18)
    badge(d, (H1[0], H1[1] + 153), ":9001 admin (allow-list)", ORANGE, P(t, c[1] + 3, 0.5), 18)
    edge_label(d, lerp(S1, CTL, 0.5), "OpenFlow 1.3 :6653", TEAL, show["e_of"])
    edge_label(d, lerp(NAT, CTL, 0.5), "REST :8080", PURPLE, show["e_rest"])
    if t > c[3] + 1:
        for k in range(30):
            flow_packets(d, t, [H2, S1, H1], [c[3] + 1 + k * 1.1], 1.4, GREEN, r=7)
    notes = [(c[1], "h1 — threaded TCP log server + policy engine", BLUE),
             (c[2], "h2 — normal, trusted client", GREEN),
             (c[2] + 1.8, "h3 — flooder / unauthorised client", RED),
             (c[3], "s1 — OVS, two-table pipeline", TEAL),
             (c[3] + 2.5, "Ryu — rule engine, stats poller, REST API", PURPLE),
             (c[4], "nat0 — bridges Mininet namespaces to the VM", MUTED)]
    for i, (te, s, col) in enumerate(notes):
        b = P(t, te, 0.5)
        if b <= 0.01:
            continue
        x = 1380 + (1 - b) * 40
        d.ellipse((x, 300 + i * 85 - 8, x + 16, 300 + i * 85 + 8), fill=mix(col, b))
        para(d, (x + 30, 300 + i * 85 - 16), s, 22, 450, FG, b)


# --- 3 channels ---------------------------------------------------------------
@scene([
    "Four conversations make the system work.",
    "First, clients talk to the log server over plain TCP sockets. The server is threaded, one thread per client, and replies ACK, BLOCKED, or DENIED.",
    "Second, the log server calls the controller's REST API, saying what to do: block this IP, or deny this flow.",
    "Third, the controller decides how, and sends OpenFlow flow-mod messages to the switch.",
    "Fourth, every second, the controller polls flow statistics from the switch, and serves them back to the log server.",
], accent=PURPLE, energy=2,
    sfx=lambda c, D: [(c[i], "tick") for i in range(1, 5)])
def s_channels(d, t, c, D):
    title(d, t, "How the pieces talk", "four channels, four protocols", PURPLE)
    act = {"h1": 0, "ctl": 0, "s1": 0}
    if c[1] <= t < c[2]:
        act["h1"] = 1
    if c[2] <= t < c[3]:
        act["ctl"] = 1
    if c[3] <= t:
        act["s1"] = 1
    topo(d, t, glow=act)
    steps = [
        (c[1], "1", "TCP sockets", "client → log server, ports 9000 / 9001; reply ACK · BLOCKED · DENIED", BLUE),
        (c[2], "2", "REST (JSON over HTTP)", "POST /firewall/block · /firewall/rule · /firewall/rule/remove", PURPLE),
        (c[3], "3", "OpenFlow 1.3", "OFPFlowMod add / strict delete into table 0", TEAL),
        (c[4], "4", "Flow statistics", "polled every 1 s → GET /firewall/stats", YELLOW),
    ]
    for i, (te, n, head, body, col) in enumerate(steps):
        a = P(t, te, 0.5)
        if a <= 0.01:
            continue
        y = 230 + i * 165
        x = 1330 + (1 - a) * 80
        nxt = steps[i + 1][0] if i < 3 else D + 10
        hi = 1.0 if te <= t < nxt else 0.55
        box(d, (x, y, x + 550, y + 145), col, a * hi)
        d.ellipse((x + 20, y + 20, x + 70, y + 70), fill=mix(col, a))
        text(d, (x + 45, y + 45), n, 28, BG, a, "bold")
        text(d, (x + 90, y + 45), head, 26, FG, a * hi, "bold", anchor="lm")
        para(d, (x + 25, y + 82), body, 19, 500, MUTED, a * hi)
    if t >= c[1]:
        tt = [c[1] + 0.3 + k * 0.6 for k in range(40)]
        flow_packets(d, t, [H2, S1, H1], [x for x in tt if x < c[2] + 1], 1.4, GREEN)
        flow_packets(d, t, [H1, S1, H2], [x + 0.9 for x in tt if x < c[2] + 1], 1.4, BLUE, r=6)
        edge_label(d, lerp(S1, H1, 0.5), "1", BLUE, P(t, c[1], 0.4), 24)
    if t >= c[2]:
        for k in range(8):
            pulse(d, t, E_REST, c[2] + 0.5 + k * 1.3, 1.1, PURPLE, 10, "POST /firewall/block" if k == 0 else None)
        edge_label(d, (NAT[0], NAT[1] + 90), "2", PURPLE, P(t, c[2], 0.4), 24)
    if t >= c[3]:
        for k in range(6):
            pulse(d, t, [CTL, S1], c[3] + 0.4 + k * 1.2, 0.9, TEAL, 10, "FlowMod" if k == 0 else None)
            ring(d, S1, t, c[3] + 1.3 + k * 1.2, TEAL, 0.7, 40, 110, 3)
        edge_label(d, (CTL[0] - 60, 410), "3", TEAL, P(t, c[3], 0.4), 24)
    if t >= c[4]:
        for k in range(12):
            t0 = c[4] + 0.4 + k * 1.0
            pulse(d, t, [CTL, S1], t0, 0.45, YELLOW, 7)
            pulse(d, t, [S1, CTL], t0 + 0.45, 0.45, YELLOW, 7)
        edge_label(d, (CTL[0] + 60, 410), "4", YELLOW, P(t, c[4], 0.4), 24)


# --- 4 rules ------------------------------------------------------------------
@scene([
    "Every connection is classified by source, destination and port, using a policy file.",
    "That classification leads to one of three rule types.",
    "Block, at priority 100, drops everything from one source IP. It's triggered by a flood: more than 8 messages in a 5-second window on the rate-limited port.",
    "Deny, at priority 200, drops just one flow, when a host that isn't on the allow-list connects to the admin port.",
    "And Allow, at priority 300, exempts trusted hosts like h2. Higher priority wins, so allow beats deny, and deny beats block.",
], accent=YELLOW, energy=2,
    sfx=lambda c, D: [(c[2], "pop"), (c[3], "pop"), (c[4], "pop"), (c[4] + 3.0, "chime")])
def s_rules(d, t, c, D):
    title(d, t, "Policy and rule types", "config/policy.json → three kinds of OpenFlow rule", YELLOW)
    code = ['{ "listen_ports": [9000, 9001],',
            '  "rate": {"window_seconds": 5,',
            '           "threshold": 8,',
            '           "recovery_seconds": 15},',
            '  "trusted_sources": ["10.0.0.2"],',
            '  "services": {',
            '    "9000": {"name": "logging",',
            '             "policy": "rate_limit"},',
            '    "9001": {"name": "admin",',
            '             "policy": "allowlist",',
            '             "allowed_sources": ["10.0.0.2"],',
            '             "deny_seconds": 30}}}']
    hl = {1: c[2], 2: c[2], 3: c[2], 9: c[3], 10: c[3], 11: c[3], 4: c[4]}
    a = P(t, c[0], 0.6)
    d.rounded_rectangle((80, 200, 760, 900), radius=14, fill=mix(LOG_FILL, 0.9 * a), outline=mix(GRID, a), width=2)
    text(d, (110, 235), "config/policy.json", 20, MUTED, a, "mono", anchor="lm")
    for i, ln in enumerate(code):
        col = FG
        te = hl.get(i)
        k = P(t, c[0] + 0.2 + i * 0.08, 0.3)
        if te is not None and t >= te:
            nxt = min([x for x in c if x > te] + [D])
            on = P(t, te, 0.4) * (1 - P(t, nxt, 0.4))
            if on > 0.01:
                d.rounded_rectangle((95, 280 + i * 50 - 22, 95 + 650 * on, 280 + i * 50 + 22), radius=6, fill=mix(YELLOW, 0.16 * on))
            col = lerpc(FG, YELLOW, on)
        text(d, (110 + (1 - k) * 20, 280 + i * 50), ln, 22, col, a * k, "mono", anchor="lm")
    cards = [
        (c[2], "BLOCK", "priority 100", "drop all traffic from one source IP",
         "trigger: > 8 msgs / 5 s on port 9000", "recovery: 15 s of silence", RED),
        (c[3], "DENY", "priority 200", "drop one flow: src, dst, TCP, port",
         "trigger: non-allow-listed host → port 9001", "recovery: removed after 30 s", ORANGE),
        (c[4], "ALLOW", "priority 300", "skip drop rules → goto table 1",
         "trigger: source in trusted_sources", "recovery: permanent while server runs", GREEN),
    ]
    for i, (te, name, prio, what, trig, rec, col) in enumerate(cards):
        b = P(t, te, 0.6)
        if b <= 0.01:
            continue
        y = 200 + i * 240
        x = 820 + (1 - back_out(b)) * 120
        box(d, (x, y, x + 1020, y + 215), col, b)
        d.rounded_rectangle((x, y, x + 10, y + 215), radius=5, fill=mix(col, b))
        text(d, (x + 34, y + 45), name, 40, col, b, "bold", anchor="lm")
        # priority meter
        pv = {"BLOCK": 100, "DENY": 200, "ALLOW": 300}[name]
        m = P(t, te + 0.4, 0.8)
        d.rounded_rectangle((x + 700, y + 37, x + 980, y + 53), radius=8, fill=mix(GRID, b * 0.6))
        d.rounded_rectangle((x + 700, y + 37, x + 700 + 280 * pv / 300 * m, y + 53), radius=8, fill=mix(col, b))
        text(d, (x + 690, y + 45), prio, 22, MUTED, b, "mono", anchor="rm")
        text(d, (x + 34, y + 100), what, 27, FG, b, anchor="lm")
        text(d, (x + 34, y + 145), trig, 23, MUTED, b, anchor="lm")
        text(d, (x + 34, y + 183), rec, 23, MUTED, b, anchor="lm")
    e = P(t, c[4] + 3.0, 0.6)
    text(d, (1330, 935), "allow 300  >  deny 200  >  block 100  >  forwarding 1", 26, YELLOW, e, "monob")


# --- 5 pipeline ---------------------------------------------------------------
@scene([
    "Inside the switch, the controller builds a two-table pipeline.",
    "Table zero is the policy table, holding the allow, deny, and block rules.",
    "Anything that isn't dropped continues to table one, where the controller's learned MAC-address forwarding rules live.",
    "So an allow rule simply means: skip the drop rules, and go straight to forwarding.",
], tail=2.5, accent=TEAL, energy=2,
    sfx=lambda c, D: [(c[1] + 3.2 + 1.6, "block"), (c[3] + 0.6 + 2.8, "chime")])
def s_pipeline(d, t, c, D):
    title(d, t, "The two-table pipeline", "controller/firewall_controller.py")
    a0 = P(t, c[1], 0.6)
    a1 = P(t, c[2], 0.6)
    box(d, (360, 220, 1080, 800), TEAL, a0)
    text(d, (720, 265), "Table 0 · policy", 34, FG, a0, "bold")
    rows = [("300", "ip, nw_src=10.0.0.2", "goto_table:1", GREEN),
            ("200", "tcp, 10.0.0.3→10.0.0.1:9001", "drop", ORANGE),
            ("100", "ip, nw_src=10.0.0.3", "drop", RED),
            ("0", "table-miss", "goto_table:1", MUTED)]
    RY = [360, 470, 580, 690]
    for i, (pr, m, act, col) in enumerate(rows):
        b = P(t, c[1] + 0.5 + i * 0.4, 0.4)
        if b <= 0.01:
            continue
        y = RY[i]
        x = (1 - b) * 40
        d.rounded_rectangle((390 + x, y - 40, 1050 + x, y + 40), radius=10, fill=mix(LOG_FILL, 0.8 * b), outline=mix(col, b), width=2)
        text(d, (420 + x, y), pr, 26, col, b, "monob", anchor="lm")
        text(d, (500 + x, y), m, 22, FG, b, "mono", anchor="lm")
        text(d, (1030 + x, y), act, 22, col, b, "monob", anchor="rm")
    box(d, (1240, 220, 1800, 800), BLUE, a1)
    text(d, (1520, 265), "Table 1 · forwarding", 34, FG, a1, "bold")
    rows1 = [("1", "eth_dst=…:01 → port 3"), ("1", "eth_dst=…:02 → port 1"), ("1", "eth_dst=…:03 → port 2"), ("0", "miss → controller")]
    for i, (pr, m) in enumerate(rows1):
        b = P(t, c[2] + 0.4 + i * 0.3, 0.4)
        if b <= 0.01:
            continue
        y = RY[i]
        x = (1 - b) * 40
        d.rounded_rectangle((1270 + x, y - 40, 1770 + x, y + 40), radius=10, fill=mix(LOG_FILL, 0.8 * b), outline=mix(BLUE, b), width=2)
        text(d, (1295 + x, y), pr, 26, BLUE, b, "monob", anchor="lm")
        text(d, (1340 + x, y), m, 21, FG, b, "mono", anchor="lm")
    arrow(d, [(1080, 510), (1240, 510)], MUTED, a1, 5, a1)
    text(d, (1160, 480), "not dropped", 18, MUTED, a1)
    text(d, (200, 510), "packet in", 24, MUTED, a0)
    arrow(d, [(140, 545), (360, 545)], MUTED, a0, 5, a0)

    def travel(t0, col, row, dropped):
        k = t - t0
        if k < 0:
            return
        if k < 0.8:
            p = lerp((140, 545), (370, 545), ease(k / 0.8))
        elif k < 1.6:
            f = ease((k - 0.8) / 0.8)
            p = (370 + 20 * f, 545 + (RY[row] - 545) * f)
        elif dropped:
            if k < 2.6:
                d.rounded_rectangle((390, RY[row] - 40, 1050, RY[row] + 40), radius=10, fill=mix(RED, 0.18 * (2.6 - k)), outline=mix(col, 2.6 - k), width=5)
                kk = (k - 1.6)
                for j in range(8):
                    ang = j * 0.785
                    q = (410 + math.cos(ang) * 60 * kk, RY[row] + math.sin(ang) * 60 * kk)
                    d.ellipse((q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5), fill=mix(RED, 1 - kk))
                text(d, (1100, RY[row]), "DROP", 26, RED, 1 - kk, "bold", anchor="lm")
            return
        elif k < 2.8:
            f = ease((k - 1.6) / 1.2)
            d.rounded_rectangle((390, RY[row] - 40, 1050, RY[row] + 40), radius=10, fill=mix(col, 0.15 * (1 - f)), outline=mix(col, 1 - f), width=5)
            p = lerp((390, RY[row]), (1290, RY[1]), f)
        elif k < 3.6:
            p = lerp((1290, RY[1]), (1860, RY[1]), ease((k - 2.8) / 0.8))
        else:
            return
        dot(d, p, col, 13)

    travel(c[1] + 3.2, RED, 2, True)
    travel(c[2] + 2.0, MUTED, 3, False)
    travel(c[3] + 0.6, GREEN, 0, False)
    travel(c[3] + 4.6, GREEN, 0, False)
    text(d, (1080, 870), "h2 matches priority 300 first → jumps to table 1, even if a drop rule also matches", 24, GREEN, P(t, c[3], 0.5))


# --- 6 scenario A -------------------------------------------------------------
def a_events(c):
    first = c[1] + 0.6
    span = (c[2] - 0.5) - first
    ok = [first + span * k / 7 for k in range(8)]
    ninth = c[2] + 0.25
    t_rest = ninth + 0.4
    t_rule = t_rest + 2.0
    t_unblock = c[4] + 5.0
    return ok, ninth, t_rest, t_rule, t_unblock


A_TRAVEL = 1.2


def a_cam(t, c, D):
    ok, ninth, t_rest, t_rule, t_unblock = a_events(c)
    z = 1.0
    sh = 0.0
    if t_rule <= t < t_rule + 0.5:
        k = 1 - (t - t_rule) / 0.5
        sh = 7 * k
    return z, (900, 560), (sh * math.sin(t * 90), sh * math.cos(t * 77))


@scene([
    "Let's watch it work. Scenario A: h3 starts flooding port 9000.",
    "The log server counts messages per source in a sliding window. The first eight are acknowledged.",
    "The ninth crosses the threshold. The server raises an alert, and posts a block request to the controller.",
    "The controller installs a priority 100 drop rule on the switch. From now on, h3's packets die at s1, before they ever reach the server.",
    "A blocked host goes quiet. So after 15 seconds of silence, the log server removes the rule, and h3 can reach the network again.",
], tail=4.5, accent=RED, energy=3, cam=a_cam,
    sfx=lambda c, D: ([(x, "tick") for x in a_events(c)[0]] +
                      [(a_events(c)[1], "alert"), (a_events(c)[3], "block"), (a_events(c)[4], "chime")]))
def s_flood(d, t, c, D):
    title(d, t, "Scenario A · flood → BLOCK → recovery", "h3$ client.py --mode flood --count 40", RED)
    ok, ninth, t_rest, t_rule, t_unblock = a_events(c)
    blocked = t_rule <= t < t_unblock
    topo(d, t, glow={"h3": 1 if c[0] < t < c[4] else 0, "s1": 1 if blocked else 0,
                     "h1": 1 if ninth <= t < t_rest + 1 else 0})
    send = [x - A_TRAVEL for x in ok + [ninth]]
    later = [ninth - A_TRAVEL + 0.32 * k for k in range(1, 200) if ninth - A_TRAVEL + 0.32 * k < c[4]]
    flow_packets(d, t, [H3, S1, H1], send + later, A_TRAVEL, RED, drop_after=t_rule)
    for x in ok:
        flow_packets(d, t, [H1, S1, H3], [x], A_TRAVEL, GREEN, r=6, trail=False)
    n = sum(1 for x in ok + [ninth] if t >= x)
    if t >= c[1]:
        a = P(t, c[1], 0.4)
        # window meter
        x0, y0 = H1[0] - 140, H1[1] + 95
        d.rounded_rectangle((x0, y0, x0 + 280, y0 + 70), radius=12, fill=mix(PANEL, 0.9 * a), outline=mix(RED if n > 8 else YELLOW, a), width=2)
        text(d, (x0 + 140, y0 + 20), "msgs in 5 s window:  %d / 8" % min(n, 9), 17, RED if n > 8 else MUTED, a, "monob")
        for i in range(9):
            col = (RED if i == 8 else YELLOW) if i < n else GRID
            bx = x0 + 22 + i * 27
            d.rounded_rectangle((bx, y0 + 38, bx + 20, y0 + 58), radius=4, fill=mix(col, a))
    ring(d, H1, t, ninth, YELLOW, 0.8, 60, 220, 5)
    pulse(d, t, E_REST, t_rest, 1.2, PURPLE, 11, "POST /firewall/block")
    pulse(d, t, [CTL, S1], t_rest + 1.25, 0.7, TEAL, 11, "FlowMod")
    ring(d, S1, t, t_rule, RED, 1.0, 50, 260, 7)
    ring(d, S1, t, t_rule + 0.25, RED, 1.0, 50, 200, 4)
    if t_rule <= t < t_rule + 0.6:
        d.rectangle((0, 0, DW, DH), fill=mix(RED, 0.16 * (1 - (t - t_rule) / 0.6)))
    ra = P(t, t_rule, 0.4) * (1 - P(t, t_unblock, 0.5))
    badge(d, (S1[0], S1[1] + 190), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, ra, 18)
    dropped = sum(1 for x in later if t >= x + A_TRAVEL * 0.5 and x + A_TRAVEL * 0.5 >= t_rule)
    if ra > 0.01:
        text(d, (S1[0], S1[1] + 235), "n_packets dropped: %d" % dropped, 20, RED, ra, "monob")
    # 'BLOCKED' stamp on h3
    st = P(t, t_rule + 0.3, 0.3) * (1 - P(t, c[4], 0.4))
    if st > 0.01:
        sc = 1.6 - 0.6 * back_out(st)
        w, h = 190 * sc, 52 * sc
        cx, cy = H3[0], H3[1] - 100
        d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), radius=8, fill=mix((60, 10, 20), 0.85 * st), outline=mix(RED, st), width=4)
        text(d, (cx, cy), "BLOCKED", 30 * sc, RED, st, "bold")
    if c[4] <= t < t_unblock + 0.6:
        q = clamp((t - c[4]) / 5.0) * 15
        qa = P(t, c[4], 0.3) * (1 - P(t, t_unblock + 0.2, 0.4))
        cx, cy = H3[0], H3[1] - 120
        d.ellipse((cx - 46, cy - 46, cx + 46, cy + 46), outline=mix(GRID, qa), width=6)
        d.arc((cx - 46, cy - 46, cx + 46, cy + 46), -90, -90 + 360 * q / 15, fill=mix(YELLOW, qa), width=6)
        text(d, (cx, cy - 6), "%.0fs" % q, 24, YELLOW, qa, "monob")
        text(d, (cx, cy + 18), "quiet", 15, MUTED, qa)
        text(d, (cx + 140, cy), "time-lapse ×3", 18, MUTED, qa)
    pulse(d, t, E_REST, t_unblock - 1.2, 1.2, PURPLE, 11, "POST /firewall/unblock")
    ring(d, S1, t, t_unblock, GREEN, 1.0, 50, 220, 5)
    flow_packets(d, t, [H3, S1, H1], [t_unblock + 0.8, t_unblock + 1.8], A_TRAVEL, GREEN)
    text(d, (H3[0] + 250, H3[1] + 85), "ping 10.0.0.1  ✓", 24, GREEN, P(t, t_unblock + 2.0, 0.4), "bold")
    entries = [(c[0] + 0.5, "Listening on 0.0.0.0:9000 (service=logging, policy=rate_limit)", MUTED),
               (c[1], "CONNECT 10.0.0.3 -> 10.0.0.1:9000 service=logging class=normal", FG)]
    entries += [(x, "EVENT 10.0.0.3 msg %d (blocked=False) → ACK" % (i + 1), FG) for i, x in enumerate(ok)]
    entries += [(ninth, "ALERT: 10.0.0.3 sent 9 msgs in 5s (threshold 8)", YELLOW),
                (t_rule, "ACTION: 10.0.0.3 BLOCKED by SDN controller", RED),
                (t_rule + 0.4, "EVENT 10.0.0.3 msg 10 (blocked=True) → BLOCKED", RED),
                (t_unblock, "ACTION: 10.0.0.3 UNBLOCKED after 15.0s quiet", GREEN)]
    log_panel(d, t, entries)


# --- 7 scenario B -------------------------------------------------------------
@scene([
    "Scenario B is more surgical. h3 connects to the admin port, 9001, which only h2 may use.",
    "The server replies DENIED, and installs a deny rule for exactly that flow: h3 to h1, TCP, port 9001, for 30 seconds.",
    "Meanwhile, h3 can still use the logging service on port 9000, and h2 can still reach the admin port.",
], tail=2.5, accent=ORANGE, energy=3,
    sfx=lambda c, D: [(c[0] + 3.2, "deny"), (c[1] + 2.3, "block")])
def s_deny(d, t, c, D):
    title(d, t, "Scenario B · unauthorised access → DENY", "h3$ client.py --port 9001", ORANGE)
    topo(d, t)
    tr = 1.2
    flow_packets(d, t, [H3, S1, H1], [c[0] + 2.0], tr, ORANGE)
    t_reply = c[0] + 2.0 + tr
    ring(d, H1, t, t_reply - 0.1, ORANGE, 0.8, 60, 200, 5)
    pulse(d, t, [H1, S1, H3], t_reply, tr, RED, 10, "DENIED")
    t_rest = c[1] + 0.3
    pulse(d, t, E_REST, t_rest, 1.2, PURPLE, 11, "POST /firewall/rule")
    pulse(d, t, [CTL, S1], t_rest + 1.25, 0.7, TEAL, 11, "FlowMod")
    t_rule = t_rest + 2.0
    ring(d, S1, t, t_rule, ORANGE, 1.0, 50, 230, 6)
    badge(d, (S1[0] - 10, S1[1] + 190), "priority=200,tcp,nw_src=10.0.0.3,", ORANGE, P(t, t_rule, 0.4), 18)
    badge(d, (S1[0] - 10, S1[1] + 235), "nw_dst=10.0.0.1,tp_dst=9001 actions=drop", ORANGE, P(t, t_rule + 0.15, 0.4), 18)
    if t >= c[2]:
        base = c[2] + 0.3
        flow_packets(d, t, [H3, S1, H1], [base + k * 1.4 for k in range(10)], tr, ORANGE, drop_after=t_rule)
        flow_packets(d, t, [H3, (S1[0], S1[1] + 22), (H1[0], H1[1] + 22)], [base + 0.7 + k * 1.4 for k in range(10)], tr, BLUE)
        flow_packets(d, t, [H2, S1, H1], [base + 0.35 + k * 1.4 for k in range(10)], tr, GREEN)
        items = [(ORANGE, "h3 → :9001", "dropped at s1", RED), (BLUE, "h3 → :9000", "ACK", GREEN), (GREEN, "h2 → :9001", "ACK", GREEN)]
        for i, (col, s, r, rc) in enumerate(items):
            leg = P(t, c[2] + i * 0.3, 0.4)
            x = 1360 + (1 - leg) * 40
            d.rounded_rectangle((x - 20, 665 + i * 70, x + 510, 715 + i * 70), radius=12, fill=mix(PANEL, 0.85 * leg), outline=mix(col, leg), width=2)
            dot(d, (x + 10, 690 + i * 70), col, 9, leg)
            text(d, (x + 40, 690 + i * 70), s, 23, FG, leg, "mono", anchor="lm")
            text(d, (x + 490, 690 + i * 70), r, 23, rc, leg, "bold", anchor="rm")
    entries = [(c[0] + 2.0 + tr, "CONNECT 10.0.0.3 -> 10.0.0.1:9001 service=admin class=violation", FG),
               (c[0] + 2.0 + tr + 0.2, "ALERT: POLICY VIOLATION 10.0.0.3 -> 10.0.0.1:9001 (10.0.0.2 is allow-listed)", YELLOW),
               (t_rule, "ACTION: DENY 10.0.0.3 -> 10.0.0.1:9001/tcp installed for 30s", ORANGE)]
    log_panel(d, t, entries, rect=(1330, 180, 1880, 620))


# --- 8 scenario C -------------------------------------------------------------
@scene([
    "Scenario C: the trusted host h2 sends the very same flood.",
    "When the log server starts, it installs a priority 300 allow rule for h2, and skips rate limiting for it. So every single message is acknowledged.",
], tail=2.5, accent=GREEN, energy=3,
    sfx=lambda c, D: [(0.4, "chime")])
def s_allow(d, t, c, D):
    title(d, t, "Scenario C · trusted host → ALLOW", "h2$ client.py --mode flood --count 40", GREEN)
    topo(d, t, glow={"h2": 1, "s1": P(t, 0.3, 0.5)})
    ring(d, S1, t, 0.4, GREEN, 1.0, 50, 230, 6)
    badge(d, (S1[0], S1[1] + 190), "table=0,priority=300,ip,nw_src=10.0.0.2", GREEN, P(t, 0.3, 0.5), 18)
    badge(d, (S1[0], S1[1] + 235), "actions=goto_table:1", GREEN, P(t, 0.45, 0.5), 18)
    # shield around h2
    sa = P(t, 0.6, 0.6)
    if sa > 0.01:
        r = 175 + 6 * math.sin(t * 3)
        d.arc((H2[0] - r, H2[1] - r * 0.62, H2[0] + r, H2[1] + r * 0.62), 0, 360, fill=mix(GREEN, 0.5 * sa), width=3)
    tr = 1.2
    sends = [c[0] + 0.6 + 0.28 * k for k in range(200) if c[0] + 0.6 + 0.28 * k < D - 1.5]
    flow_packets(d, t, [H2, S1, H1], sends, tr, GREEN)
    flow_packets(d, t, [H1, S1, H2], [x + tr for x in sends], tr, BLUE, r=5, trail=False)
    n = sum(1 for x in sends if t >= x + tr)
    badge(d, (H1[0], H1[1] + 110), "h2 replies: ACK × %d   BLOCKED × 0" % n, GREEN, P(t, c[0] + 1, 0.4), 20)
    entries = [(0.4, "ACTION: ALLOW rule for trusted source 10.0.0.2 installed on the controller", GREEN),
               (c[0] + 0.6 + tr, "CONNECT 10.0.0.2 -> 10.0.0.1:9000 service=logging class=trusted", FG)]
    entries += [(x + tr, "EVENT 10.0.0.2 msg %d (blocked=False) → ACK" % (i + 1), FG) for i, x in enumerate(sends)]
    log_panel(d, t, entries)


# --- 9 scenario D -------------------------------------------------------------
@scene([
    "What if the controller crashes?",
    "The log server has a circuit breaker. It logs one warning, keeps serving clients without stalling, and retries every five seconds.",
    "A restarted controller has lost its rule table. So every two seconds, a reconcile loop compares the controller's rules with what the server expects, and re-installs anything missing.",
    "The network heals itself, with no operator involved.",
], tail=2.0, accent=PURPLE, energy=3,
    sfx=lambda c, D: [(c[0] + 0.5, "down"), (c[2] + 0.6, "up"), (c[2] + 3.9, "block"), (c[3], "chime")])
def s_robust(d, t, c, D):
    title(d, t, "Scenario D · controller failure and recovery", "circuit breaker + reconcile loop", PURPLE)
    down = c[0] + 0.5 <= t < c[2] + 0.6
    topo(d, t, ctl_dim=down, glow={"ctl": P(t, c[2] + 0.6, 0.4) * (1 - P(t, c[2] + 2, 0.5))})
    if down:
        k = P(t, c[0] + 0.5, 0.3)
        cross(d, CTL, RED, 44 * back_out(k), k, 9)
        text(d, (CTL[0], CTL[1] - 88), "Ctrl+C", 22, RED, k, "bold")
        # sparks
        for j in range(5):
            ph = (t * 1.7 + j * 0.37) % 1
            ang = j * 1.3 + 0.5
            q = (CTL[0] + 150 + math.cos(ang) * 40 * ph, CTL[1] - 50 + math.sin(ang) * 40 * ph - 30 * ph)
            d.ellipse((q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3), fill=mix(YELLOW, 1 - ph))
    ring(d, CTL, t, c[0] + 0.5, RED, 0.8, 60, 220, 6)
    tr = 1.2
    if t >= c[1]:
        sends = [c[1] + 0.5 + 0.5 * k for k in range(40) if c[1] + 0.5 + 0.5 * k < c[2]]
        flow_packets(d, t, [H3, S1, H1], sends, tr, RED)
        for k in range(4):
            t0 = c[1] + 1.2 + k * 2.2
            if t0 < c[2]:
                f = (t - t0) / 0.9
                if 0 <= f <= 1:
                    dot(d, path_point([H1, NAT, lerp(NAT, CTL, 0.5)], f), PURPLE, 9)
                elif 1 < f < 1.5:
                    cross(d, lerp(NAT, CTL, 0.5), RED, 14, 1 - (f - 1) * 2, 5)
        badge(d, (H1[0], H1[1] + 110), "serving · every reply ACK", BLUE, P(t, c[1], 0.4) * (1 - P(t, c[2], 0.4)), 20)
        cb = P(t, c[1], 0.4) * (1 - P(t, c[2] + 0.6, 0.4))
        badge(d, (NAT[0], NAT[1] + 100), "circuit breaker: OPEN", YELLOW, cb, 18)
    ring(d, CTL, t, c[2] + 0.6, PURPLE, 0.9, 60, 220, 6)
    if t >= c[2]:
        empty = 1 - P(t, c[2] + 3.6, 0.3)
        badge(d, (CTL[0] - 330, CTL[1]), "rules: 0", MUTED, P(t, c[2] + 0.6, 0.3) * empty, 20)
        for k in range(3):
            pulse(d, t, E_REST, c[2] + 1.2 + k * 2.0, 1.0, PURPLE, 10, "GET /firewall/rules" if k == 0 else None)
        pulse(d, t, E_REST, c[2] + 2.4, 1.0, PURPLE, 10, "POST /firewall/rule")
        pulse(d, t, [CTL, S1], c[2] + 3.3, 0.6, TEAL, 11, "FlowMod ×2")
        ra = P(t, c[2] + 3.9, 0.4)
        ring(d, S1, t, c[2] + 3.9, GREEN, 1.0, 50, 230, 6)
        badge(d, (S1[0], S1[1] + 190), "priority=300,ip,nw_src=10.0.0.2 goto_table:1", GREEN, ra, 18)
        badge(d, (S1[0], S1[1] + 235), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, P(t, c[2] + 4.05, 0.4), 18)
        badge(d, (CTL[0] - 330, CTL[1]), "rules: 2", GREEN, ra, 20)
    if t >= c[3]:
        k = P(t, c[3], 0.5)
        text(d, (720, 120 + 800), "✓ healed automatically", 36 * (0.8 + 0.2 * back_out(k)), GREEN, k, "bold")
    entries = [(c[1], "WARNING: controller 10.0.0.254:8080 unreachable; continuing without blocking, retry in 5s", YELLOW),
               (c[1] + 1.5, "EVENT 10.0.0.3 msg 12 (blocked=False) → ACK", FG),
               (c[2] + 1.6, "controller 10.0.0.254:8080 reachable again", GREEN),
               (c[2] + 3.6, "ACTION: ALLOW rule for trusted source 10.0.0.2 installed on the controller", GREEN),
               (c[2] + 3.8, "ACTION: BLOCK rule for 10.0.0.3 installed on the controller", RED)]
    log_panel(d, t, entries)


# --- 10 scenario E ------------------------------------------------------------
def e_times(c):
    t_cross = c[2] + 3.0
    return c[1] + 0.5, t_cross, t_cross + 2.2


@scene([
    "Detection doesn't have to come from the application alone.",
    "The controller's statistics poller turns per-flow byte counters into per-source throughput.",
    "With bandwidth detection enabled, a source measured above a threshold, say 3 megabits per second, is blocked on the switch's own evidence.",
], tail=2.5, accent=YELLOW, energy=3,
    sfx=lambda c, D: [(e_times(c)[1], "alert"), (e_times(c)[2], "block")])
def s_bw(d, t, c, D):
    title(d, t, "Scenario E · detection from switch statistics", "log_server.py --bw-threshold-mbps 3", YELLOW)
    t_start, t_cross, t_block = e_times(c)
    topo(d, t, glow={"h3": 1 if t_start < t < t_block else 0})
    for k in range(40):
        t0 = c[1] + 0.3 + k * 0.9
        if t0 < D:
            pulse(d, t, [CTL, S1], t0, 0.4, YELLOW, 7)
            pulse(d, t, [S1, CTL], t0 + 0.4, 0.4, YELLOW, 7)
    a = P(t, c[1], 0.5)
    X0, Y0, X1, Y1 = 1360, 230, 1860, 700
    d.rounded_rectangle((X0 - 30, Y0 - 50, X1 + 20, Y1 + 80), radius=14, fill=mix(LOG_FILL, 0.9 * a), outline=mix(GRID, a), width=2)
    text(d, (X0, Y0 - 22), "h3 throughput measured at s1 (Mbit/s)", 20, MUTED, a, anchor="lm")
    ymax = 8.0
    ty = lambda v: Y1 - (v / ymax) * (Y1 - Y0)
    for v in (0, 2, 4, 6, 8):
        d.line((X0, ty(v), X1, ty(v)), fill=mix(GRID, a * 0.6), width=1)
        text(d, (X0 - 8, ty(v)), str(v), 16, MUTED, a, anchor="rm")
    thr = P(t, c[2], 0.5)
    d.line((X0, ty(3), X0 + (X1 - X0) * thr, ty(3)), fill=mix(YELLOW, thr * a), width=3)
    text(d, (X1, ty(3) - 16), "threshold 3 Mbps", 18, YELLOW, thr * a, anchor="rm")

    def val(tt):
        if tt < t_start:
            return 0
        if tt < t_block:
            return min(7.2, 0.2 + (tt - t_start) * 6.5 / max(0.1, t_cross - t_start + 0.6)) + 0.25 * math.sin(tt * 7)
        return max(0.0, 7.2 - (tt - t_block) * 9)
    span = D - t_start
    pts, tt = [], t_start
    while tt <= t:
        pts.append((X0 + (tt - t_start) / span * (X1 - X0), ty(clamp(val(tt), 0, ymax))))
        tt += 1 / FPS
    if len(pts) > 1:
        poly = pts + [(pts[-1][0], ty(0)), (pts[0][0], ty(0))]
        d.polygon(poly, fill=mix(RED, 0.15 * a))
        d.line(pts, fill=mix(RED, a), width=4)
        dot(d, pts[-1], RED, 7, a)
    text(d, (X0 + 250, Y1 + 40), "illustrative trace", 16, MUTED, a)
    sends = [c[1] + 0.3 + k * 0.18 for k in range(400) if c[1] + 0.3 + k * 0.18 < D]
    flow_packets(d, t, [H3, S1, H1], sends, 1.2, RED, r=8, drop_after=t_block, trail=False)
    pulse(d, t, E_REST, t_cross + 0.4, 1.0, PURPLE, 11, "POST /firewall/block")
    pulse(d, t, [CTL, S1], t_cross + 1.45, 0.6, TEAL, 11, None)
    ring(d, S1, t, t_block, RED, 1.0, 50, 260, 7)
    ring(d, (X0 + (t_cross - t_start) / span * (X1 - X0), ty(3)), t, t_cross, YELLOW, 0.8, 10, 90, 4)
    badge(d, (S1[0], S1[1] + 190), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, P(t, t_block, 0.4), 18)
    para(d, (X0 - 20, Y1 + 100), "ALERT: 10.0.0.3 measured 7.2 Mbps by the switch → BLOCKED", 20, 520, YELLOW, P(t, t_cross, 0.4), "mono")


# --- 11 evaluation ------------------------------------------------------------
@scene([
    "To measure the benefit, an experiment harness compares three conditions: idle; a static baseline, where the log server only monitors; and the dynamic firewall.",
    "Each runs at low, medium, and high flood intensity, measuring the legitimate host's throughput, latency and loss, plus the attacker's accepted traffic, time to block, and packets dropped.",
    "The harness and analysis scripts are written and tested; running them on Mininet is the next step.",
    "Unit tests and a localhost end-to-end test cover the rule engine, the policy server, and real OpenFlow message serialisation.",
], accent=BLUE, energy=2,
    sfx=lambda c, D: [(c[0] + 0.6 + i * 0.6, "pop") for i in range(3)] + [(c[3] + i * 0.5, "tick") for i in range(4)])
def s_eval(d, t, c, D):
    title(d, t, "Evaluation and testing", "experiments/run_experiments.py · experiments/analyze.py · tests/", BLUE)
    conds = [("idle", "no flood (reference)", MUTED), ("static", "flood, monitor-only (--no-block)", ORANGE), ("dynamic", "flood, SDN block active", GREEN)]
    ints = [("low", "1 conn · 20 msg/s", 1), ("medium", "1 conn · unthrottled", 2), ("high", "4 conns · saturates 10 Mbit/s", 3)]
    GX, GY = 100, 230
    cw, ch = 280, 110
    for j, (n, s, lv) in enumerate(ints):
        b = P(t, c[1] + j * 0.3, 0.4)
        cx = GX + 330 + j * cw + cw / 2
        text(d, (cx, GY - 10), n, 28, FG, b, "bold")
        text(d, (cx, GY + 24), s, 17, MUTED, b)
        for k in range(3):   # intensity bars
            col = RED if k < lv else GRID
            d.rounded_rectangle((cx + 50 + k * 12, GY - 8 - k * 7, cx + 58 + k * 12, GY + 4), radius=2, fill=mix(col, b))
    for i, (n, s, col) in enumerate(conds):
        b = P(t, c[0] + 0.6 + i * 0.6, 0.4)
        y = GY + 70 + i * ch
        text(d, (GX + (1 - b) * -30, y + ch / 2 - 14), n, 30, col, b, "bold", anchor="lm")
        text(d, (GX, y + ch / 2 + 20), s, 18, MUTED, b, anchor="lm")
        for j in range(3):
            bb = min(b, P(t, c[1] + j * 0.3 + i * 0.1, 0.4)) if i else b * P(t, c[1], 0.4) * (1 if j == 0 else 0.35)
            if bb <= 0.01:
                continue
            sc = back_out(bb)
            cx, cy = GX + 330 + j * cw + cw / 2, y + ch / 2
            box(d, (cx - (cw - 20) / 2 * sc, cy - (ch - 16) / 2 * sc, cx + (cw - 20) / 2 * sc, cy + (ch - 16) / 2 * sc), col, bb, r=12, w=2)
            text(d, (cx, cy), "3 runs × 20 s", 20 * sc, MUTED, bb)
    m = P(t, c[1] + 1.5, 0.5)
    text(d, (GX, 670), "Measured for h2 (legitimate)", 24, GREEN, m, "bold", anchor="lm")
    text(d, (GX, 710), "iperf throughput · ping RTT · loss", 22, FG, m, anchor="lm")
    text(d, (GX + 520, 670), "Measured for h3 (attacker)", 24, RED, m, "bold", anchor="lm")
    text(d, (GX + 520, 710), "accepted msgs · time to first block · switch drops", 22, FG, m, anchor="lm")
    s = P(t, c[2], 0.5)
    box(d, (GX, 760, GX + 1100, 840), YELLOW, s, r=12, w=2)
    text(d, (GX + 550, 800), "Status: harness + parsers tested on sample data · Mininet runs pending", 22, YELLOW, s)
    tests = [("test_controller_logic.py", "rule engine, priorities, strict removal, real OFP serialisation"),
             ("test_policy_server.py", "classification, threshold, deny, controller-down, bandwidth"),
             ("test_experiment_parsing.py", "ping / iperf / client parsers"),
             ("local_test.py", "real Ryu + log server + clients on localhost")]
    for i, (n, s2) in enumerate(tests):
        b = P(t, c[3] + i * 0.5, 0.4)
        if b <= 0.01:
            continue
        y = 260 + i * 140
        x = 1320 + (1 - b) * 60
        box(d, (x, y, x + 540, y + 120), BLUE, b)
        ck = P(t, c[3] + i * 0.5 + 0.3, 0.3)
        d.ellipse((x + 18, y + 22, x + 46, y + 50), fill=mix(GREEN, ck))
        text(d, (x + 32, y + 36), "✓", 20, BG, ck, "bold")
        text(d, (x + 58, y + 36), n, 21, GREEN, b, "monob", anchor="lm")
        para(d, (x + 25, y + 62), s2, 19, 490, MUTED, b)


# --- 12 end -------------------------------------------------------------------
@scene([
    "To sum up: the application detects, the controller decides, and the switch enforces.",
    "Block for misbehaving hosts, deny for unauthorised flows, allow for trusted ones; all installed and removed at run time. That's a dynamic firewall.",
], tail=4.0, accent=TEAL, energy=1,
    sfx=lambda c, D: [(c[0] + i * 1.1, "pop") for i in range(3)] + [(c[1] + 4.5, "chime")])
def s_end(d, t, c, D):
    steps = [("Application", "detects", "TCP log server · sliding window · policy", BLUE, "server"),
             ("Controller", "decides", "Ryu · REST API · rule engine", PURPLE, "gear"),
             ("Switch", "enforces", "OVS · OpenFlow 1.3 · table 0 drops", TEAL, "switch")]
    for i, (n, v, s, col, ic) in enumerate(steps):
        a = P(t, c[0] + i * 1.1, 0.6)
        if a > 0.01:
            sc = back_out(a)
            x = 330 + i * 630
            box(d, (x - 260 * sc, 375 - 165 * sc, x + 260 * sc, 375 + 165 * sc), col, a)
            d.ellipse((x - 40, 220, x + 40, 300), fill=mix(col, 0.15 * a))
            icon(d, ic, x, 260, col, a, t, 1.3 * sc)
            text(d, (x, 340), n, 44 * sc, FG, a, "bold")
            text(d, (x, 405), v, 38 * sc, col, a, "bold")
            text(d, (x, 475), s, 21 * sc, MUTED, a)
        if i < 2:
            b = P(t, c[0] + i * 1.1 + 0.6, 0.5)
            x = 330 + i * 630
            arrow(d, [(x + 270, 375), (x + 360, 375)], MUTED, b, 6, b)
            if b > 0.5:
                dot(d, (x + 270 + 90 * ((t * 0.8) % 1), 375), YELLOW, 6, b)
    rules = [("BLOCK", "misbehaving hosts", RED), ("DENY", "unauthorised flows", ORANGE), ("ALLOW", "trusted sources", GREEN)]
    for i, (n, s, col) in enumerate(rules):
        a = P(t, c[1] + i * 0.8, 0.5)
        x = 330 + i * 630
        text(d, (x, 620 + (1 - a) * 30), n, 42, col, a, "bold")
        text(d, (x, 672 + (1 - a) * 30), s, 26, FG, a)
    e = P(t, c[1] + 4.5, 0.8)
    text(d, (DW / 2, 790), "installed and removed at run time — no operator in the loop", 32, YELLOW, e, "bold")
    f = P(t, D - 2.6, 0.8)
    text(d, (DW / 2, 860), "github · dynamic-firewall  ·  Mininet + Ryu + Python sockets", 22, MUTED, f, "mono")


# ================================================================ background
class Background:
    def __init__(self, scale):
        self.s = scale
        W, H = int(DW * scale), int(DH * scale)
        BW, BH = int(W * 1.12), int(H * 1.12)
        yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
        u, v = xx / BW, yy / BH
        top = np.array([12, 16, 34], np.float32)
        bot = np.array([18, 12, 38], np.float32)
        img = top[None, None, :] * (1 - v[..., None]) + bot[None, None, :] * v[..., None]
        # soft blobs
        for (cx, cy, r, col, k) in ((0.18, 0.25, 0.35, (40, 70, 160), 0.55), (0.85, 0.7, 0.4, (90, 40, 150), 0.45),
                                    (0.6, 0.1, 0.25, (20, 110, 130), 0.3)):
            dd = ((u - cx) ** 2 + ((v - cy) * BH / BW) ** 2) / (r * r)
            img += np.exp(-dd * 2.2)[..., None] * (np.array(col, np.float32)[None, None, :] * k)
        # vignette
        vd = ((u - 0.5) ** 2 + (v - 0.5) ** 2)
        img *= (1 - 0.55 * vd)[..., None]
        # dot grid
        step = 48 * scale
        gx = (np.abs(((xx + step / 2) % step) - step / 2) < 1.2 * scale)
        gy = (np.abs(((yy + step / 2) % step) - step / 2) < 1.2 * scale)
        img[gx & gy] += 22
        # film grain
        rng = np.random.default_rng(3)
        img += rng.normal(0, 2.0, img.shape[:2])[..., None]
        self.big = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
        self.W, self.H, self.BW, self.BH = W, H, BW, BH
        self.glows = {}

    def glow(self, col):
        if col not in self.glows:
            n = int(900 * self.s)
            yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
            dd = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (n / 2) ** 2
            alpha = np.clip(np.exp(-dd * 3.2) * 70, 0, 255).astype(np.uint8)
            rgb = np.zeros((n, n, 3), np.uint8)
            rgb[...] = col
            self.glows[col] = (Image.fromarray(rgb, "RGB"), Image.fromarray(alpha, "L"))
        return self.glows[col]

    def frame(self, ta, accent):
        ox = (self.BW - self.W) / 2 + math.sin(ta * 0.07) * (self.BW - self.W) * 0.45
        oy = (self.BH - self.H) / 2 + math.cos(ta * 0.05) * (self.BH - self.H) * 0.45
        img = self.big.crop((int(ox), int(oy), int(ox) + self.W, int(oy) + self.H))
        rgb, alpha = self.glow(accent)
        n = rgb.size[0]
        gx = int(self.W * (0.75 + 0.12 * math.sin(ta * 0.21)) - n / 2)
        gy = int(self.H * (0.30 + 0.15 * math.cos(ta * 0.17)) - n / 2)
        img.paste(rgb, (gx, gy), alpha)
        gx = int(self.W * (0.2 + 0.1 * math.cos(ta * 0.13)) - n / 2)
        gy = int(self.H * (0.8 + 0.1 * math.sin(ta * 0.19)) - n / 2)
        img.paste(rgb, (gx, gy), alpha)
        return img


PARTICLES = [(random.Random(i).uniform(0, DW), random.Random(i + 99).uniform(0, DH),
              random.Random(i + 7).uniform(4, 16), random.Random(i + 3).uniform(1.0, 2.6)) for i in range(55)]


def particles(d, ta):
    for (x, y, sp, r) in PARTICLES:
        yy = (y - ta * sp) % DH
        xx = x + 12 * math.sin(ta * 0.3 + y)
        d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(170, 200, 255, 50))


# ================================================================ captions
def caption(d, s, prog, a=1.0):
    if not s or a <= 0.01:
        return
    size = 30
    lines = wrap_lines(s, size, 1560)
    lh = 42
    y0 = 1040 - lh * len(lines)
    width = max(font(size).getlength(l) for l in lines) + 70
    d.rounded_rectangle((DW / 2 - width / 2, y0 - 22, DW / 2 + width / 2, 1050), radius=20,
                        fill=(6, 9, 18, int(200 * a)), outline=(80, 96, 130, int(90 * a)), width=1)
    total = sum(len(l) for l in lines)
    spoken = prog * total * 1.03
    done = 0
    f = font(size)
    for i, ln in enumerate(lines):
        x = DW / 2 - f.getlength(ln) / 2
        for wd in ln.split(" "):
            on = clamp((spoken - done) / max(1, len(wd)))
            col = lerpc((120, 130, 152), (250, 252, 255), on)
            d.text((x, y0 + i * lh + 12), wd, font=f, fill=col + (int(255 * a),), anchor="lm")
            x += f.getlength(wd + " ")
            done += len(wd) + 1


# ================================================================ audio
def synth_voice(sentences, cache_dir):
    from kokoro_onnx import Kokoro
    k = None
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for s in sentences:
        h = hashlib.md5(("%s|%s|%s" % (VOICE, SPEED, s)).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, h + ".wav")
        if not os.path.exists(path):
            if k is None:
                k = Kokoro(os.path.join(HERE, "voice/kokoro.onnx"), os.path.join(HERE, "voice/voices.bin"))
            samples, sr = k.create(s, voice=VOICE, speed=SPEED, lang="en-us")
            sf.write(path, samples, sr)
        data, sr = sf.read(path, dtype="float32")
        assert sr == VOICE_SR
        # upsample 24k -> 48k
        x = np.arange(len(data) * 2) / 2.0
        out.append(np.interp(x, np.arange(len(data)), data).astype(np.float32))
    return out


def note_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def env_adsr(n, a, r, sr=SR):
    e = np.ones(n, np.float32)
    na, nr = max(1, int(a * sr)), max(1, int(r * sr))
    e[:na] = np.linspace(0, 1, na)
    e[-nr:] *= np.linspace(1, 0, nr)
    return e


def music(total, energy_at, seed=1):
    """Ambient electronic bed: pad + bass + arpeggio + drums, layered by energy (1..3)."""
    n = int(total * SR) + SR
    out = np.zeros(n, np.float32)
    bpm = 96
    beat = 60 / bpm
    bar = beat * 4
    # Am  F  C  G   (midi roots)
    prog = [(57, [57, 60, 64, 69]), (53, [53, 57, 60, 65]), (48, [55, 60, 64, 67]), (55, [55, 59, 62, 67])]
    t_axis = None
    nbars = int(total / bar) + 2
    rng = np.random.default_rng(seed)
    for b in range(nbars):
        t0 = b * bar
        root, chord = prog[b % 4]
        e = energy_at(t0 + bar / 2)
        # pad: detuned additive saw-ish, slow attack
        L = int((bar + 1.2) * SR)
        tt = np.arange(L) / SR
        pad = np.zeros(L, np.float32)
        for m in chord:
            for det in (-0.12, 0.12):
                f = note_hz(m) * 2 ** (det / 12)
                for h in range(1, 6):
                    pad += (0.5 / h) * np.sin(2 * np.pi * f * h * tt + h * 0.7).astype(np.float32)
        pad *= env_adsr(L, 0.8, 1.2) * 0.010
        i0 = int(t0 * SR)
        out[i0:i0 + L] += pad[:max(0, min(L, n - i0))]
        # bass: eighth notes
        if e >= 1.5:
            for k in range(8):
                ts = t0 + k * beat / 2
                Lb = int(beat / 2 * 0.95 * SR)
                tb = np.arange(Lb) / SR
                f = note_hz(root - 12)
                sig = (np.sin(2 * np.pi * f * tb) + 0.35 * np.sin(4 * np.pi * f * tb)) * np.exp(-tb * 5)
                sig *= env_adsr(Lb, 0.005, 0.03) * 0.07 * (1 if k % 2 == 0 else 0.7)
                j = int(ts * SR)
                out[j:j + Lb] += sig[:max(0, min(Lb, n - j))].astype(np.float32)
        elif e >= 1.0:
            Lb = int(bar * SR)
            tb = np.arange(Lb) / SR
            f = note_hz(root - 12)
            sig = np.sin(2 * np.pi * f * tb) * env_adsr(Lb, 0.3, 0.6) * 0.05
            j = int(t0 * SR)
            out[j:j + Lb] += sig[:max(0, min(Lb, n - j))].astype(np.float32)
        # arpeggio: 16ths
        if e >= 2.0:
            pattern = [0, 1, 2, 3, 2, 1, 2, 3]
            for k in range(16):
                ts = t0 + k * beat / 4
                m = chord[pattern[k % 8]] + 12
                La = int(0.35 * SR)
                ta = np.arange(La) / SR
                f = note_hz(m)
                sig = (np.sin(2 * np.pi * f * ta) + 0.15 * np.sin(6 * np.pi * f * ta)) * np.exp(-ta * 11)
                sig *= 0.022 * (1.0 if k % 4 == 0 else 0.7)
                j = int(ts * SR)
                out[j:j + La] += sig[:max(0, min(La, n - j))].astype(np.float32)
        # drums
        if e >= 2.5:
            for k in range(4):
                ts = t0 + k * beat
                Lk = int(0.3 * SR)
                tk = np.arange(Lk) / SR
                freq = 45 + 90 * np.exp(-tk * 30)
                ph = 2 * np.pi * np.cumsum(freq) / SR
                sig = np.sin(ph) * np.exp(-tk * 12) * 0.16
                j = int(ts * SR)
                out[j:j + Lk] += sig[:max(0, min(Lk, n - j))].astype(np.float32)
        if e >= 2.0:
            for k in range(8):
                ts = t0 + k * beat / 2 + (beat / 4 if False else 0)
                if k % 2 == 0 and e < 2.5:
                    continue
                Lh = int(0.06 * SR)
                noise = rng.standard_normal(Lh).astype(np.float32)
                noise = np.diff(noise, prepend=0)  # crude high-pass
                sig = noise * np.exp(-np.arange(Lh) / SR * 70) * (0.012 if k % 2 else 0.007)
                j = int(ts * SR)
                out[j:j + Lh] += sig[:max(0, min(Lh, n - j))]
    # simple stereo-less reverb-ish smear: feedback delay
    d1 = int(0.23 * SR)
    for _ in range(3):
        out[d1:] += out[:-d1] * 0.18
    return out


def sfx_sample(kind, rng):
    def tt(sec):
        return np.arange(int(sec * SR)) / SR
    if kind == "whoosh":
        t = tt(0.8)
        noise = rng.standard_normal(len(t)).astype(np.float32)
        # moving-average lowpass with varying width
        cs = np.cumsum(noise)
        w = (6 + 40 * (1 - np.sin(np.pi * t / 0.8))).astype(int)
        idx = np.arange(len(t))
        lo = np.clip(idx - w, 0, None)
        sig = (cs - cs[lo]) / w
        return sig * np.sin(np.pi * t / 0.8) ** 2 * 0.18
    if kind == "boom":
        t = tt(1.6)
        f = 38 + 60 * np.exp(-t * 6)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2) * 0.35
        return sig
    if kind == "pop":
        t = tt(0.12)
        f = 700 + 900 * t / 0.12
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 35) * 0.10
    if kind == "tick":
        t = tt(0.05)
        return np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 90) * 0.07
    if kind == "alert":
        t = tt(0.14)
        a = np.sin(2 * np.pi * 880 * t) * env_adsr(len(t), 0.005, 0.03)
        b = np.sin(2 * np.pi * 1175 * t) * env_adsr(len(t), 0.005, 0.03)
        gap = np.zeros(int(0.04 * SR))
        return np.concatenate([a, gap, b, gap, a, gap, b]) * 0.09
    if kind == "block":
        t = tt(0.6)
        f = 40 + 140 * np.exp(-t * 18)
        thud = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 0.4
        noise = rng.standard_normal(len(t)) * np.exp(-t * 25) * 0.08
        zap = np.sign(np.sin(2 * np.pi * (300 - 250 * t / 0.6) * t)) * np.exp(-t * 14) * 0.04
        return thud + noise + zap
    if kind == "chime":
        parts = []
        for i, m in enumerate((84, 88, 91)):
            t = tt(1.4)
            f = note_hz(m)
            s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)) * np.exp(-t * 3.5)
            pad = np.zeros(int(i * 0.09 * SR))
            parts.append(np.concatenate([pad, s]))
        L = max(len(p) for p in parts)
        out = np.zeros(L)
        for p in parts:
            out[:len(p)] += p
        return out * 0.05
    if kind == "deny":
        t = tt(0.32)
        return np.sign(np.sin(2 * np.pi * 180 * t)) * env_adsr(len(t), 0.005, 0.06) * 0.05
    if kind == "down":
        t = tt(0.9)
        f = 700 * np.exp(-t * 3) + 60
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_adsr(len(t), 0.01, 0.3) * 0.12
    if kind == "up":
        t = tt(0.6)
        f = 200 + 800 * (t / 0.6) ** 2
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * env_adsr(len(t), 0.01, 0.2) * 0.09
    raise ValueError(kind)


# ================================================================ build
def build_timeline():
    all_s = [s for sc in SCENES for s in sc["s"]]
    clips = synth_voice(all_s, os.path.join(HERE, "tts_cache2"))
    timeline, spans, k, t_abs = [], [], 0, 0.0
    for sc in SCENES:
        cues, t = [], LEAD
        for s in sc["s"]:
            dur = len(clips[k]) / SR
            cues.append(t)
            spans.append((t_abs + t, t_abs + t + dur, s, k))
            t += dur + GAP
            k += 1
        D = t - GAP + sc["tail"]
        timeline.append(dict(start=t_abs, D=D, cues=cues, **sc))
        t_abs += D
    return timeline, spans, clips, t_abs


def scene_at(timeline, ta):
    for i, sc in enumerate(timeline):
        if ta < sc["start"] + sc["D"] or i == len(timeline) - 1:
            return sc


def render_frame(ta, timeline, spans, bgobj, scale):
    sc = scene_at(timeline, ta)
    t, D = ta - sc["start"], sc["D"]
    img = bgobj.frame(ta, sc["accent"])
    base = Cam(img, scale)
    particles(base, ta)
    # camera: slow push + transitions
    zoom = 1.0 + 0.012 * (t / D)
    zoom += 0.06 * (1 - ease(t / 0.7))           # arrive from slightly zoomed in
    zoom -= 0.03 * ease((t - (D - 0.5)) / 0.5)   # leave by pulling back
    focus, shake = (DW / 2, DH / 2), (0, 0)
    if sc["cam"]:
        z2, focus, shake = sc["cam"](t, sc["cues"], D)
        zoom *= z2
    d = Cam(img, scale, zoom, focus, shake)
    sc["fn"](d, t, sc["cues"], D)
    cap, prog = None, 0
    for (st, en, s, _) in spans:
        if st - 0.1 <= ta <= en + GAP * 0.9:
            cap, prog = s, clamp((ta - st) / max(0.1, en - st))
    caption(base, cap, prog)
    fade = min(clamp(t / 0.3), clamp((D - t) / 0.3))
    if ta < 0.5:
        fade = clamp(ta / 0.5)
    if fade < 1:
        img = Image.blend(Image.new("RGB", img.size, (0, 0, 0)), img, fade)
    return img


def worker(idx, f0, f1, seg_path, height, timeline_args):
    scale = height / DH
    timeline, spans = timeline_args
    bgobj = Background(scale)
    W, H = int(DW * scale), int(DH * scale)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-threads", "2", seg_path], stdin=subprocess.PIPE)
    t_start = time.time()
    for fi in range(f0, f1):
        img = render_frame(fi / FPS, timeline, spans, bgobj, scale)
        ff.stdin.write(img.tobytes())
        if (fi - f0) % 300 == 0:
            el = time.time() - t_start
            print("[w%d] %d/%d  %.1f fps" % (idx, fi - f0, f1 - f0, (fi - f0 + 1) / max(el, 1e-3)), flush=True)
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview", type=int)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    timeline, spans, clips, total = build_timeline()
    print("total %.1fs, %d scenes" % (total, len(timeline)), flush=True)
    scale = a.height / DH
    if a.preview is not None or a.bench:
        bgobj = Background(scale)
        if a.bench:
            sc = timeline[6]
            t0 = time.time()
            for k in range(20):
                render_frame(sc["start"] + sc["cues"][3] + k * 0.1, timeline, spans, bgobj, scale)
            print("%.3f s/frame at %dp" % ((time.time() - t0) / 20, a.height))
            return
        sc = timeline[a.preview]
        for frac in (0.15, 0.4, 0.65, 0.9):
            img = render_frame(sc["start"] + sc["D"] * frac, timeline, spans, bgobj, scale)
            img.save(a.out.replace(".mp4", "_s%d_%d.png" % (a.preview, int(frac * 100))))
        return
    # ---- audio
    n = int(total * SR) + SR
    voice = np.zeros(n, np.float32)
    for (st, en, s, idx) in spans:
        i0 = int(st * SR)
        voice[i0:i0 + len(clips[idx])] += clips[idx]

    def energy_at(tt):
        sc = scene_at(timeline, tt)
        return sc["energy"]
    bed = music(total, energy_at)[:n]
    # duck music under speech
    gate = np.zeros(n, np.float32)
    for (st, en, s, idx) in spans:
        gate[int(st * SR):int(en * SR)] = 1
    k = int(0.25 * SR)
    ker = np.ones(k, np.float32) / k
    gate = np.convolve(gate, ker, mode="same")
    duck = 1.0 - 0.55 * gate
    fx = np.zeros(n, np.float32)
    rng = np.random.default_rng(5)
    events = []
    for sc in timeline:
        events.append((sc["start"], "whoosh"))
        events += [(sc["start"] + te, kind) for te, kind in sc["sfx"](sc["cues"], sc["D"])]
    for te, kind in events:
        s = np.asarray(sfx_sample(kind, rng), np.float32)
        i0 = int(max(0, te) * SR)
        L = max(0, min(len(s), n - i0))
        fx[i0:i0 + L] += s[:L]
    # global fade-out of the bed at the end
    fo = int(3.0 * SR)
    bed[-fo:] *= np.linspace(1, 0, fo)
    bed[:int(1.0 * SR)] *= np.linspace(0, 1, int(1.0 * SR))
    mixd = voice * 1.0 + bed * duck * 1.6 + fx * 1.0
    mixd = np.tanh(mixd * 1.1) / np.tanh(1.1)
    mixd *= 0.95 / max(1e-6, np.max(np.abs(mixd)))
    wav = a.out + ".wav"
    sf.write(wav, mixd, SR)
    # subtitles
    def ts(x):
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return ("%02d:%02d:%06.3f" % (h, m, s)).replace(".", ",")
    with open(a.out.replace(".mp4", ".srt"), "w") as f:
        for i, (st, en, s, _) in enumerate(spans):
            f.write("%d\n%s --> %s\n%s\n\n" % (i + 1, ts(st), ts(en), s))
    # ---- video segments in parallel
    nframes = int(total * FPS)
    bounds = [int(nframes * i / a.workers) for i in range(a.workers + 1)]
    segs = [a.out + ".seg%d.mp4" % i for i in range(a.workers)]
    procs = []
    lite = ([{k: v for k, v in sc.items()} for sc in timeline], spans)
    for i in range(a.workers):
        p = mp.get_context("fork").Process(target=worker, args=(i, bounds[i], bounds[i + 1], segs[i], a.height, lite))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        assert p.exitcode == 0, "worker failed"
    lst = a.out + ".txt"
    with open(lst, "w") as f:
        for s in segs:
            f.write("file '%s'\n" % os.path.abspath(s))
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                           "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", a.out])
    for s in segs + [lst, wav]:
        os.remove(s)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
