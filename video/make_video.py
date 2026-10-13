"""Explainer video for the Dynamic Firewall project.

Narration: Kokoro TTS, one clip per sentence. Visuals: Pillow frames keyed to sentence cues.
Usage: python3 make_video.py OUT.mp4 [--preview SCENE_INDEX]
"""
import math, os, subprocess, sys, textwrap
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, SR = 1920, 1080, 30, 24000
LEAD, GAP, TAIL = 0.6, 0.45, 1.0

# ---------------------------------------------------------------- palette
BG = (14, 20, 34)
PANEL = (24, 33, 52)
GRID = (40, 52, 78)
FG = (232, 238, 248)
MUTED = (140, 155, 182)
BLUE = (88, 160, 255)      # h1 server
GREEN = (72, 205, 130)     # h2 trusted
RED = (255, 96, 96)        # h3 attacker / drop
ORANGE = (255, 170, 70)    # deny
PURPLE = (176, 132, 255)   # controller / REST
TEAL = (64, 210, 210)      # switch / OpenFlow
YELLOW = (255, 214, 90)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
_fonts = {}


def font(size, kind="sans"):
    key = (size, kind)
    if key not in _fonts:
        name = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf",
                "mono": "DejaVuSansMono.ttf", "monob": "DejaVuSansMono-Bold.ttf"}[kind]
        _fonts[key] = ImageFont.truetype(FONT_DIR + name, size)
    return _fonts[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def P(t, start, dur=0.5):
    """Eased progress 0..1 of an animation starting at `start`."""
    if dur <= 0:
        return 1.0 if t >= start else 0.0
    return ease((t - start) / dur)


def mix(c, a, base=BG):
    return tuple(int(base[i] + (c[i] - base[i]) * a) for i in range(3))


def lerp(p, q, f):
    return (p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f)


# ---------------------------------------------------------------- primitives
def text(d, xy, s, size=32, col=FG, a=1.0, kind="sans", anchor="mm", base=BG):
    if a <= 0.01:
        return
    d.text(xy, s, font=font(size, kind), fill=mix(col, a, base), anchor=anchor)


def box(d, rect, col, a=1.0, fill=PANEL, r=18, w=3, base=BG):
    if a <= 0.01:
        return
    d.rounded_rectangle(rect, radius=r, fill=mix(fill, a, base), outline=mix(col, a, base), width=w)


def line(d, pts, col, a=1.0, w=4, frac=1.0, base=BG):
    """Draw a polyline, optionally only the first `frac` of its length."""
    if a <= 0.01 or frac <= 0:
        return
    pts = path_prefix(pts, frac)
    d.line(pts, fill=mix(col, a, base), width=w, joint="curve")


def path_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def path_point(pts, f):
    f = clamp(f)
    total = path_len(pts)
    target = total * f
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


def dot(d, p, col, r=11, a=1.0, base=BG):
    if a <= 0.01:
        return
    x, y = p
    d.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4), fill=mix(col, 0.25 * a, base))
    d.ellipse((x - r, y - r, x + r, y + r), fill=mix(col, a, base))


def cross(d, p, col=RED, s=18, a=1.0, w=6):
    if a <= 0.01:
        return
    x, y = p
    c = mix(col, a)
    d.line((x - s, y - s, x + s, y + s), fill=c, width=w)
    d.line((x - s, y + s, x + s, y - s), fill=c, width=w)


def arrowhead(d, p, q, col, a=1.0, s=16):
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    pts = [q, (q[0] - s * math.cos(ang - 0.45), q[1] - s * math.sin(ang - 0.45)),
           (q[0] - s * math.cos(ang + 0.45), q[1] - s * math.sin(ang + 0.45))]
    d.polygon(pts, fill=mix(col, a))


def arrow(d, pts, col, a=1.0, w=5, frac=1.0):
    if a <= 0.01 or frac <= 0:
        return
    pp = path_prefix(pts, frac)
    d.line(pp, fill=mix(col, a), width=w, joint="curve")
    if len(pp) >= 2 and math.dist(pp[-2], pp[-1]) > 2:
        arrowhead(d, pp[-2], pp[-1], col, a)


def wrap_lines(s, size, maxw, kind="sans"):
    f = font(size, kind)
    words, lines, cur = s.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if f.getlength(trial) <= maxw:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def para(d, xy, s, size, maxw, col=FG, a=1.0, kind="sans", spacing=1.35, base=BG):
    x, y = xy
    for i, ln in enumerate(wrap_lines(s, size, maxw, kind)):
        text(d, (x, y + i * size * spacing), ln, size, col, a, kind, anchor="la", base=base)


def title(d, t, s, sub=None):
    a = P(t, 0, 0.6)
    text(d, (80, 70), s, 46, FG, a, "bold", anchor="lm")
    if sub:
        text(d, (80, 118), sub, 26, MUTED, a, anchor="lm")
    d.line((80, 148, 80 + 260 * a, 148), fill=mix(TEAL, a), width=4)


# ---------------------------------------------------------------- topology
H2, H3, S1, H1, CTL, NAT = (270, 380), (270, 800), (720, 590), (1150, 590), (720, 230), (1150, 230)
NODE_SIZE = {"host": (270, 120), "switch": (210, 110), "ctl": (300, 120), "nat": (200, 84)}

E_H2 = [H2, S1]
E_H3 = [H3, S1]
E_H1 = [S1, H1]
E_OF = [S1, CTL]
E_REST = [H1, NAT, CTL]


def node(d, c, kind, label, sub, col, a=1.0, dim=False):
    if a <= 0.01:
        return
    w, h = NODE_SIZE[kind]
    x, y = c
    rect = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
    edge = MUTED if dim else col
    fill = (30, 32, 40) if dim else PANEL
    r = 50 if kind == "switch" else 18
    box(d, rect, edge, a, fill=fill, r=r, w=4)
    # little icon band
    d.rounded_rectangle((x - w / 2 + 14, y - h / 2 + 14, x - w / 2 + 26, y + h / 2 - 14),
                        radius=6, fill=mix(edge, a))
    text(d, (x + 6, y - 18), label, 30, FG if not dim else MUTED, a, "bold")
    text(d, (x + 6, y + 22), sub, 20, MUTED, a)


def topo(d, t, show=None, ctl_dim=False, alpha=None):
    """Static topology. `show` maps element -> alpha (default all 1)."""
    s = show or {}
    g = lambda k: s.get(k, 1.0) * (alpha if alpha is not None else 1.0)
    line(d, E_H2, GRID, g("e_h2"), 6)
    line(d, E_H3, GRID, g("e_h3"), 6)
    line(d, E_H1, GRID, g("e_h1"), 6)
    line(d, E_OF, mix(TEAL, 0.45), g("e_of"), 4)
    line(d, E_REST, mix(PURPLE, 0.45), g("e_rest"), 4)
    node(d, H2, "host", "h2", "10.0.0.2 · trusted", GREEN, g("h2"))
    node(d, H3, "host", "h3", "10.0.0.3 · attacker", RED, g("h3"))
    node(d, H1, "host", "h1", "10.0.0.1 · log server", BLUE, g("h1"))
    node(d, S1, "switch", "s1", "Open vSwitch", TEAL, g("s1"))
    node(d, CTL, "ctl", "Ryu", "SDN controller", PURPLE, g("ctl"), dim=ctl_dim)
    node(d, NAT, "nat", "nat0", "10.0.0.254", MUTED, g("nat"))


def edge_label(d, p, s, col, a=1.0, size=20):
    if a <= 0.01:
        return
    f = font(size, "bold")
    w = f.getlength(s) + 20
    x, y = p
    d.rounded_rectangle((x - w / 2, y - 16, x + w / 2, y + 16), radius=10, fill=mix(BG, 1), outline=mix(col, a), width=2)
    text(d, p, s, size, col, a, "bold")


def flow_packets(d, t, pts, times, travel, col, r=10, drop_after=None, drop_at=0.5, a=1.0):
    """Packets sent at `times` travel along pts. If the packet reaches the drop
    point (fraction drop_at of the path) after time drop_after, it dies there."""
    for t0 in times:
        f = (t - t0) / travel
        if f < 0:
            continue
        dies = drop_after is not None and t0 + travel * drop_at >= drop_after
        if dies:
            if f <= drop_at:
                dot(d, path_point(pts, f), col, r, a)
            elif f <= drop_at + 0.35:
                k = 1 - (f - drop_at) / 0.35
                cross(d, (path_point(pts, drop_at)[0] - 36, path_point(pts, drop_at)[1] - 30), RED, 14, k * a, 5)
        elif f <= 1:
            dot(d, path_point(pts, f), col, r, a)


def pulse(d, t, pts, t0, travel, col, r=9, label=None):
    f = (t - t0) / travel
    if 0 <= f <= 1:
        p = path_point(pts, f)
        dot(d, p, col, r)
        if label:
            text(d, (p[0], p[1] - 30), label, 18, col, 1, "bold")


def badge(d, xy, s, col, a=1.0, size=19, anchor="mm"):
    if a <= 0.01:
        return
    f = font(size, "mono")
    w = f.getlength(s) + 28
    x, y = xy
    if anchor == "lm":
        x += w / 2
    d.rounded_rectangle((x - w / 2, y - 20, x + w / 2, y + 20), radius=10,
                        fill=mix((40, 20, 24) if col == RED else PANEL, a), outline=mix(col, a), width=3)
    text(d, (x, y), s, size, col, a, "mono")


def log_panel(d, t, entries, rect=(1330, 180, 1880, 900), title_s="logs/security_events.log", maxl=None):
    """entries: list of (time, text, color). Shows the newest lines that fit."""
    x0, y0, x1, y1 = rect
    box(d, rect, GRID, 1, fill=(18, 25, 40), r=14, w=2)
    text(d, (x0 + 20, y0 + 26), title_s, 18, MUTED, 1, "mono", anchor="lm")
    d.line((x0, y0 + 50, x1, y0 + 50), fill=GRID, width=2)
    lines = []
    for (te, s, col) in entries:
        if t >= te:
            a = P(t, te, 0.25)
            for k, ln in enumerate(wrap_lines(s, 19, x1 - x0 - 40, "mono")):
                lines.append((("  " if k else "") + ln, col, a))
    lh = 30
    cap = maxl or int((y1 - y0 - 70) / lh)
    lines = lines[-cap:]
    for i, (ln, col, a) in enumerate(lines):
        text(d, (x0 + 20, y0 + 74 + i * lh), ln, 19, col, a, "mono", anchor="lm", base=(18, 25, 40))


def caption(d, s, a):
    if not s or a <= 0.01:
        return
    lines = wrap_lines(s, 30, 1640)
    lh = 40
    y0 = 1048 - lh * len(lines)
    d.rectangle((0, y0 - 18, W, H), fill=(8, 12, 22))
    for i, ln in enumerate(lines):
        text(d, (W / 2, y0 + i * lh + 12), ln, 30, (225, 230, 240), a, base=(8, 12, 22))


# ================================================================ scenes
# Each scene: (sentences, draw(d, t, c, D), min_tail)
SCENES = []


def scene(sentences, tail=TAIL):
    def deco(fn):
        SCENES.append((sentences, fn, tail))
        return fn
    return deco


@scene([
    "This is the Dynamic Firewall with Security Event Logger: a software-defined networking project built with Mininet, the Ryu controller, and Python TCP sockets.",
    "In the next few minutes, we'll see what it does, and how it does it.",
], tail=1.4)
def s_title(d, t, c, D):
    a = P(t, 0.1, 0.9)
    text(d, (W / 2, 380), "Dynamic Firewall", 96, FG, a, "bold")
    text(d, (W / 2, 480), "with Security Event Logger", 52, TEAL, a, "bold")
    tags = [("Mininet", BLUE), ("Ryu SDN controller", PURPLE), ("OpenFlow 1.3", TEAL), ("Python TCP sockets", GREEN)]
    x = W / 2 - 600
    for i, (s, col) in enumerate(tags):
        b = P(t, 0.8 + i * 0.25, 0.5)
        wdt = font(28, "bold").getlength(s) + 50
        box(d, (x, 580, x + wdt, 640), col, b, r=30)
        text(d, (x + wdt / 2, 610), s, 28, col, b, "bold")
        x += wdt + 30
    text(d, (W / 2, 730), "SDN + TCP socket programming mini project · UE25CS243A, Project #1", 28, MUTED, P(t, 2, 0.8))


@scene([
    "A traditional firewall is static. Its rules are written in advance, and it can't react to what the application is actually seeing.",
    "This project closes that gap. A TCP log server watches its own clients, and when one misbehaves, it asks an SDN controller to change the switch's rules while the network is running.",
    "And when the misbehaviour stops, the rules are removed again, automatically.",
])
def s_idea(d, t, c, D):
    title(d, t, "The idea", "Let the application steer the network at run time")
    # left: static
    a = P(t, c[0], 0.6)
    box(d, (110, 220, 880, 880), MUTED, a)
    text(d, (495, 275), "Static firewall", 38, FG, a, "bold")
    rules = ["allow tcp any -> 10.0.0.1:9000", "deny  tcp any -> 10.0.0.1:23", "allow icmp any -> any"]
    for i, r in enumerate(rules):
        text(d, (170, 370 + i * 60), r, 26, MUTED, a, "mono", anchor="lm")
    text(d, (495, 600), "written once, by hand", 30, MUTED, a)
    # flood passes straight through
    if a > 0.5:
        for k in range(6):
            f = ((t * 0.6 + k / 6) % 1)
            dot(d, (170 + f * 650, 720), RED, 10, a * (1 - abs(f - 0.5) * 0.6))
        text(d, (495, 790), "a flood sails through until someone notices", 26, RED, a)
    # right: dynamic loop
    b = P(t, c[1], 0.6)
    box(d, (1040, 220, 1810, 880), TEAL, b)
    text(d, (1425, 275), "Dynamic firewall", 38, FG, b, "bold")
    pts = {"app": (1425, 400), "ctl": (1650, 640), "sw": (1200, 640)}
    labels = {"app": ("Log server", "detects", BLUE), "ctl": ("Controller", "decides", PURPLE), "sw": ("Switch", "enforces", TEAL)}
    order = ["app", "ctl", "sw"]
    for i, k in enumerate(order):
        bb = P(t, c[1] + 0.8 + i * 0.6, 0.5)
        x, y = pts[k]
        box(d, (x - 120, y - 55, x + 120, y + 55), labels[k][2], bb)
        text(d, (x, y - 15), labels[k][0], 28, FG, bb, "bold")
        text(d, (x, y + 20), labels[k][1], 22, labels[k][2], bb)
    for i in range(3):
        p, q = pts[order[i]], pts[order[(i + 1) % 3]]
        bb = P(t, c[1] + 2.5 + i * 0.5, 0.6)
        sp, sq = lerp(p, q, 0.32), lerp(p, q, 0.68)
        arrow(d, [sp, sq], MUTED, bb, 4, bb)
    ra = P(t, c[2], 0.6)
    text(d, (1425, 800), "rules expire on their own when traffic calms down", 26, GREEN, ra)
    if ra > 0:
        ang = t * 2.0
        for k in range(3):
            q = ang + k * 2.094
            dot(d, (1425 + 70 * math.cos(q), 560 + 40 * math.sin(q)), YELLOW, 7, ra)


@scene([
    "Here is the network, emulated in Mininet.",
    "Host h1 runs the log server, listening on two TCP ports: 9000 for logging, and 9001 for a protected admin service.",
    "Host h2 is a well-behaved, trusted client. Host h3 plays the attacker: it floods the server, or touches ports it shouldn't.",
    "All three hosts connect through a single Open vSwitch, s1, which speaks OpenFlow 1.3 to the Ryu controller.",
    "Because Mininet hosts live in isolated namespaces, a NAT node at 10.0.0.254 lets h1 reach the controller's REST API.",
])
def s_topo(d, t, c, D):
    title(d, t, "The topology", "topology/firewall_topo.py")
    show = {"h1": P(t, c[1], 0.5), "h2": P(t, c[2], 0.5), "h3": P(t, c[2] + 1.8, 0.5),
            "s1": P(t, c[3], 0.5), "e_h1": P(t, c[3] + 0.3, 0.5), "e_h2": P(t, c[3] + 0.3, 0.5),
            "e_h3": P(t, c[3] + 0.3, 0.5), "ctl": P(t, c[3] + 2.5, 0.5), "e_of": P(t, c[3] + 3, 0.5),
            "nat": P(t, c[4], 0.5), "e_rest": P(t, c[4] + 1.5, 0.6)}
    topo(d, t, show)
    a = show["h1"]
    badge(d, (H1[0], H1[1] + 100), ":9000 logging (rate-limited)", BLUE, a, 18)
    badge(d, (H1[0], H1[1] + 148), ":9001 admin (allow-list)", ORANGE, P(t, c[1] + 3, 0.5), 18)
    edge_label(d, lerp(S1, CTL, 0.5), "OpenFlow 1.3 :6653", TEAL, show["e_of"])
    edge_label(d, lerp(NAT, CTL, 0.5), "REST :8080", PURPLE, show["e_rest"])
    # side notes
    notes = [(c[1], "h1 — threaded TCP log server + policy engine", BLUE),
             (c[2], "h2 — normal, trusted client", GREEN),
             (c[2] + 1.8, "h3 — flooder / unauthorised client", RED),
             (c[3], "s1 — OVS, two-table pipeline", TEAL),
             (c[3] + 2.5, "Ryu — rule engine, stats poller, REST API", PURPLE),
             (c[4], "nat0 — bridges Mininet namespaces to the VM", MUTED)]
    for i, (te, s, col) in enumerate(notes):
        b = P(t, te, 0.5)
        d.ellipse((1380, 300 + i * 80 - 8, 1396, 300 + i * 80 + 8), fill=mix(col, b))
        para(d, (1410, 300 + i * 80 - 16), s, 22, 450, FG, b)


@scene([
    "Four conversations make the system work.",
    "First, clients talk to the log server over plain TCP sockets. The server is threaded, one thread per client, and replies ACK, BLOCKED, or DENIED.",
    "Second, the log server calls the controller's REST API, saying what to do: block this IP, or deny this flow.",
    "Third, the controller decides how, and sends OpenFlow flow-mod messages to the switch.",
    "Fourth, every second, the controller polls flow statistics from the switch, and serves them back to the log server.",
])
def s_channels(d, t, c, D):
    title(d, t, "How the pieces talk", "four channels, four protocols")
    topo(d, t)
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
        box(d, (1330, y, 1880, y + 145), col, a)
        d.ellipse((1350, y + 20, 1400, y + 70), fill=mix(col, a, PANEL))
        text(d, (1375, y + 45), n, 28, BG, a, "bold", base=PANEL)
        text(d, (1420, y + 45), head, 26, FG, a, "bold", anchor="lm", base=PANEL)
        para(d, (1355, y + 82), body, 19, 500, MUTED, a, base=PANEL)
    # animations
    if t >= c[1]:
        tt = [c[1] + 0.3 + k * 0.7 for k in range(40)]
        flow_packets(d, t, [H2, S1, H1], [x for x in tt if x < c[2] + 1], 1.4, GREEN)
        edge_label(d, lerp(S1, H1, 0.5), "1", BLUE, P(t, c[1], 0.4), 24)
    if t >= c[2]:
        for k in range(8):
            pulse(d, t, E_REST, c[2] + 0.5 + k * 1.3, 1.1, PURPLE, 10, "block" if k == 0 else None)
        edge_label(d, (NAT[0] + 0, NAT[1] + 90), "2", PURPLE, P(t, c[2], 0.4), 24)
    if t >= c[3]:
        for k in range(6):
            pulse(d, t, [CTL, S1], c[3] + 0.4 + k * 1.2, 0.9, TEAL, 10, "FlowMod" if k == 0 else None)
        edge_label(d, (CTL[0] - 60, 410), "3", TEAL, P(t, c[3], 0.4), 24)
    if t >= c[4]:
        for k in range(12):
            t0 = c[4] + 0.4 + k * 1.0
            pulse(d, t, [CTL, S1], t0, 0.45, YELLOW, 7)
            pulse(d, t, [S1, CTL], t0 + 0.45, 0.45, YELLOW, 7)
        edge_label(d, (CTL[0] + 60, 410), "4", YELLOW, P(t, c[4], 0.4), 24)


@scene([
    "Every connection is classified by source, destination and port, using a policy file.",
    "That classification leads to one of three rule types.",
    "Block, at priority 100, drops everything from one source IP. It's triggered by a flood: more than 8 messages in a 5-second window on the rate-limited port.",
    "Deny, at priority 200, drops just one flow, when a host that isn't on the allow-list connects to the admin port.",
    "And Allow, at priority 300, exempts trusted hosts like h2. Higher priority wins, so allow beats deny, and deny beats block.",
])
def s_rules(d, t, c, D):
    title(d, t, "Policy and rule types", "config/policy.json → three kinds of OpenFlow rule")
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
    box(d, (80, 200, 760, 900), GRID, a, fill=(18, 25, 40))
    text(d, (110, 235), "config/policy.json", 20, MUTED, a, "mono", anchor="lm", base=(18, 25, 40))
    active = None
    for k, te in sorted(hl.items(), key=lambda kv: kv[1]):
        pass
    for i, ln in enumerate(code):
        col = FG
        te = hl.get(i)
        if te is not None and t >= te:
            nxt = min([x for x in c if x > te] + [D])
            on = P(t, te, 0.4) * (1 - P(t, nxt, 0.4))
            if on > 0.01:
                d.rectangle((95, 280 + i * 50 - 22, 745, 280 + i * 50 + 22), fill=mix((50, 60, 30), on, (18, 25, 40)))
            col = mix(YELLOW, on, FG) if on > 0.01 else FG
        text(d, (110, 280 + i * 50), ln, 22, col, a, "mono", anchor="lm", base=(18, 25, 40))
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
        y = 200 + i * 240
        x = 820 + (1 - b) * 60
        box(d, (x, y, x + 1020, y + 215), col, b)
        text(d, (x + 30, y + 45), name, 40, col, b, "bold", anchor="lm", base=PANEL)
        text(d, (x + 990, y + 45), prio, 26, MUTED, b, "mono", anchor="rm", base=PANEL)
        text(d, (x + 30, y + 100), what, 27, FG, b, anchor="lm", base=PANEL)
        text(d, (x + 30, y + 145), trig, 23, MUTED, b, anchor="lm", base=PANEL)
        text(d, (x + 30, y + 183), rec, 23, MUTED, b, anchor="lm", base=PANEL)
    # priority ladder note
    e = P(t, c[4] + 3.0, 0.6)
    text(d, (1330, 935), "allow 300  >  deny 200  >  block 100  >  forwarding 1", 26, YELLOW, e, "monob")


@scene([
    "Inside the switch, the controller builds a two-table pipeline.",
    "Table zero is the policy table, holding the allow, deny, and block rules.",
    "Anything that isn't dropped continues to table one, where the controller's learned MAC-address forwarding rules live.",
    "So an allow rule simply means: skip the drop rules, and go straight to forwarding.",
], tail=2.5)
def s_pipeline(d, t, c, D):
    title(d, t, "The two-table pipeline", "controller/firewall_controller.py")
    a0 = P(t, c[1], 0.6)
    a1 = P(t, c[2], 0.6)
    T0 = (360, 220, 1080, 800)
    T1 = (1240, 220, 1800, 800)
    box(d, T0, TEAL, a0)
    text(d, (720, 265), "Table 0 · policy", 34, FG, a0, "bold", base=PANEL)
    rows = [("300", "ip, nw_src=10.0.0.2", "goto_table:1", GREEN),
            ("200", "tcp, 10.0.0.3→10.0.0.1:9001", "drop", ORANGE),
            ("100", "ip, nw_src=10.0.0.3", "drop", RED),
            ("0", "table-miss", "goto_table:1", MUTED)]
    RY = [360, 470, 580, 690]
    for i, (pr, m, act, col) in enumerate(rows):
        b = P(t, c[1] + 0.5 + i * 0.4, 0.4)
        y = RY[i]
        d.rounded_rectangle((390, y - 40, 1050, y + 40), radius=10, fill=mix((20, 28, 45), b, PANEL), outline=mix(col, b, PANEL), width=2)
        text(d, (420, y), pr, 26, col, b, "monob", anchor="lm", base=PANEL)
        text(d, (500, y), m, 22, FG, b, "mono", anchor="lm", base=PANEL)
        text(d, (1030, y), act, 22, col, b, "monob", anchor="rm", base=PANEL)
    box(d, T1, BLUE, a1)
    text(d, (1520, 265), "Table 1 · forwarding", 34, FG, a1, "bold", base=PANEL)
    rows1 = [("1", "eth_dst=…:01 → port 3"), ("1", "eth_dst=…:02 → port 1"), ("1", "eth_dst=…:03 → port 2"), ("0", "miss → controller")]
    for i, (pr, m) in enumerate(rows1):
        b = P(t, c[2] + 0.4 + i * 0.3, 0.4)
        y = RY[i]
        d.rounded_rectangle((1270, y - 40, 1770, y + 40), radius=10, fill=mix((20, 28, 45), b, PANEL), outline=mix(BLUE, b, PANEL), width=2)
        text(d, (1295, y), pr, 26, BLUE, b, "monob", anchor="lm", base=PANEL)
        text(d, (1340, y), m, 21, FG, b, "mono", anchor="lm", base=PANEL)
    # arrow between tables
    arrow(d, [(1080, 510), (1240, 510)], MUTED, a1, 5, a1)
    text(d, (1160, 480), "not dropped", 18, MUTED, a1)
    text(d, (200, 510), "packet in", 24, MUTED, a0)
    arrow(d, [(140, 545), (360, 545)], MUTED, a0, 5, a0)

    def travel(t0, col, row, dropped):
        """Packet enters, scans rows top-down to `row`, then drops or jumps to table 1."""
        k = t - t0
        if k < 0:
            return
        if k < 0.8:
            p = lerp((140, 545), (370, 545), ease(k / 0.8))
        elif k < 1.6:
            p = lerp((370, 545), (360, RY[row]), ease((k - 0.8) / 0.8))
            p = (370 + 20 * ease((k - 0.8) / 0.8), p[1])
        elif dropped:
            if k < 2.6:
                p = (390, RY[row])
                d.rounded_rectangle((390, RY[row] - 40, 1050, RY[row] + 40), radius=10, outline=mix(col, 1 - (k - 1.6)), width=5)
                cross(d, (1080, RY[row]), RED, 22, 1 - abs(k - 2.1) / 0.6)
            return
        elif k < 2.8:
            f = ease((k - 1.6) / 1.2)
            d.rounded_rectangle((390, RY[row] - 40, 1050, RY[row] + 40), radius=10, outline=mix(col, 1 - f), width=5)
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
    leg = P(t, c[3], 0.5)
    text(d, (1080, 870), "h2 packet matches priority 300 first → jumps to table 1 even if h2 is blocked", 24, GREEN, leg)


def a_events(c):
    """Arrival schedule for scenario A (shared by drawing and logs)."""
    first = c[1] + 0.6
    span = (c[2] - 0.5) - first
    ok = [first + span * k / 7 for k in range(8)]
    ninth = c[2] + 0.25
    t_alert = ninth
    t_rest = t_alert + 0.4
    t_rule = t_rest + 2.0
    return ok, ninth, t_alert, t_rest, t_rule


A_TRAVEL = 1.2


@scene([
    "Let's watch it work. Scenario A: h3 starts flooding port 9000.",
    "The log server counts messages per source in a sliding window. The first eight are acknowledged.",
    "The ninth crosses the threshold. The server raises an alert, and posts a block request to the controller.",
    "The controller installs a priority 100 drop rule on the switch, and from now on, h3's packets die at s1, before they ever reach the server.",
    "A blocked host goes quiet. So after 15 seconds of silence, the log server removes the rule, and h3 can reach the network again.",
], tail=4.5)
def s_flood(d, t, c, D):
    title(d, t, "Scenario A · flood → BLOCK → recovery", "h3: client.py --mode flood --count 40")
    topo(d, t)
    ok, ninth, t_alert, t_rest, t_rule = a_events(c)
    send = [x - A_TRAVEL for x in ok + [ninth]]
    # flood after the 9th, until the last cue
    later = [ninth - A_TRAVEL + 0.35 * k for k in range(1, 200) if ninth - A_TRAVEL + 0.35 * k < c[4]]
    flow_packets(d, t, [H3, S1, H1], send + later, A_TRAVEL, RED, drop_after=t_rule)
    # message counter at h1
    n = sum(1 for x in ok + [ninth] if t >= x)
    if t >= c[1]:
        col = RED if n > 8 else YELLOW
        badge(d, (H1[0], H1[1] + 110), "msgs in 5 s window: %d / 8" % n, col, P(t, c[1], 0.4), 20)
    # REST + OpenFlow
    pulse(d, t, E_REST, t_rest, 1.2, PURPLE, 11, "POST /firewall/block")
    pulse(d, t, [CTL, S1], t_rest + 1.25, 0.7, TEAL, 11, "FlowMod")
    t_unblock = c[4] + 5.0
    ra = P(t, t_rule, 0.4) * (1 - P(t, t_unblock, 0.5))
    badge(d, (S1[0], S1[1] + 190), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, ra, 18)
    dropped = sum(1 for x in later if t >= x + A_TRAVEL * 0.5 and x + A_TRAVEL * 0.5 >= t_rule)
    if ra > 0.01:
        text(d, (S1[0], S1[1] + 235), "n_packets dropped: %d" % dropped, 20, RED, ra, "mono")
    # quiet timer
    if t >= c[4] and t < t_unblock + 0.6:
        q = clamp((t - c[4]) / 5.0) * 15
        badge(d, (H3[0], H3[1] - 105), "quiet: %4.1f s / 15" % q, YELLOW, P(t, c[4], 0.3) * (1 - P(t, t_unblock + 0.2, 0.4)), 20)
        if t >= c[4]:
            text(d, (H3[0], H3[1] - 150), "time-lapse ×3", 18, MUTED, P(t, c[4], 0.3) * (1 - P(t, t_unblock + 0.2, 0.4)))
    pulse(d, t, E_REST, t_unblock - 1.2, 1.2, PURPLE, 11, "POST /firewall/unblock")
    # ping works again
    flow_packets(d, t, [H3, S1, H1], [t_unblock + 0.8, t_unblock + 1.8], A_TRAVEL, GREEN)
    text(d, (H3[0] + 220, H3[1] + 85), "ping 10.0.0.1 ✓", 22, GREEN, P(t, t_unblock + 2.0, 0.4), "bold")
    entries = [(c[0] + 0.5, "Listening on 0.0.0.0:9000 (service=logging, policy=rate_limit)", MUTED),
               (c[1], "CONNECT 10.0.0.3 -> 10.0.0.1:9000 service=logging class=normal", FG)]
    entries += [(x, "EVENT 10.0.0.3 msg %d (blocked=False) → ACK" % (i + 1), FG) for i, x in enumerate(ok)]
    entries += [(t_alert, "ALERT: 10.0.0.3 sent 9 msgs in 5s (threshold 8)", YELLOW),
                (t_rule, "ACTION: 10.0.0.3 BLOCKED by SDN controller", RED),
                (t_rule + 0.4, "EVENT 10.0.0.3 msg 10 (blocked=True) → BLOCKED", RED),
                (t_unblock, "ACTION: 10.0.0.3 UNBLOCKED after 15.0s quiet", GREEN)]
    log_panel(d, t, entries)


@scene([
    "Scenario B is more surgical. h3 connects to the admin port, 9001, which only h2 may use.",
    "The server replies DENIED, and installs a deny rule for exactly that flow: h3 to h1, TCP, port 9001, for 30 seconds.",
    "Meanwhile, h3 can still use the logging service on port 9000, and h2 can still reach the admin port.",
], tail=2.5)
def s_deny(d, t, c, D):
    title(d, t, "Scenario B · unauthorised access → DENY", "h3: client.py --port 9001")
    topo(d, t)
    tr = 1.2
    flow_packets(d, t, [H3, S1, H1], [c[0] + 2.0], tr, ORANGE)
    t_reply = c[0] + 2.0 + tr
    pulse(d, t, [H1, S1, H3], t_reply, tr, RED, 10, "DENIED")
    t_rest = c[1] + 0.3
    pulse(d, t, E_REST, t_rest, 1.2, PURPLE, 11, "POST /firewall/rule")
    pulse(d, t, [CTL, S1], t_rest + 1.25, 0.7, TEAL, 11, "FlowMod")
    t_rule = t_rest + 2.0
    badge(d, (S1[0] - 10, S1[1] + 190), "priority=200,tcp,nw_src=10.0.0.3,", ORANGE, P(t, t_rule, 0.4), 18)
    badge(d, (S1[0] - 10, S1[1] + 235), "nw_dst=10.0.0.1,tp_dst=9001 actions=drop", ORANGE, P(t, t_rule, 0.4), 18)
    if t >= c[2]:
        base = c[2] + 0.3
        flow_packets(d, t, [H3, S1, H1], [base + k * 1.4 for k in range(10)], tr, ORANGE, drop_after=t_rule)
        flow_packets(d, t, [H3, (S1[0], S1[1] + 22), (H1[0], H1[1] + 22)], [base + 0.7 + k * 1.4 for k in range(10)], tr, BLUE)
        flow_packets(d, t, [H2, S1, H1], [base + 0.35 + k * 1.4 for k in range(10)], tr, GREEN)
        leg = P(t, c[2], 0.4)
        items = [(ORANGE, "h3 → :9001  dropped at s1"), (BLUE, "h3 → :9000  ACK"), (GREEN, "h2 → :9001  ACK")]
        for i, (col, s) in enumerate(items):
            dot(d, (1380, 690 + i * 55), col, 10, leg)
            text(d, (1410, 690 + i * 55), s, 24, FG, leg, anchor="lm")
    entries = [(c[0] + 2.0 + tr, "CONNECT 10.0.0.3 -> 10.0.0.1:9001 service=admin class=violation", FG),
               (c[0] + 2.0 + tr + 0.2, "ALERT: POLICY VIOLATION 10.0.0.3 -> 10.0.0.1:9001 (10.0.0.2 is allow-listed)", YELLOW),
               (t_rule, "ACTION: DENY 10.0.0.3 -> 10.0.0.1:9001/tcp installed for 30s", ORANGE)]
    log_panel(d, t, entries, rect=(1330, 180, 1880, 620))


@scene([
    "Scenario C: the trusted host h2 sends the very same flood.",
    "When the log server starts, it installs a priority 300 allow rule for h2, and skips rate limiting for it, so every single message is acknowledged.",
], tail=2.5)
def s_allow(d, t, c, D):
    title(d, t, "Scenario C · trusted host → ALLOW", "h2: client.py --mode flood --count 40")
    topo(d, t)
    badge(d, (S1[0], S1[1] + 190), "table=0,priority=300,ip,nw_src=10.0.0.2", GREEN, P(t, 0.3, 0.5), 18)
    badge(d, (S1[0], S1[1] + 235), "actions=goto_table:1", GREEN, P(t, 0.3, 0.5), 18)
    tr = 1.2
    sends = [c[0] + 0.6 + 0.3 * k for k in range(200) if c[0] + 0.6 + 0.3 * k < D - 1.5]
    flow_packets(d, t, [H2, S1, H1], sends, tr, GREEN)
    n = sum(1 for x in sends if t >= x + tr)
    badge(d, (H1[0], H1[1] + 110), "h2 replies: ACK × %d   BLOCKED × 0" % n, GREEN, P(t, c[0] + 1, 0.4), 20)
    entries = [(0.4, "ACTION: ALLOW rule for trusted source 10.0.0.2 installed on the controller", GREEN),
               (c[0] + 0.6 + tr, "CONNECT 10.0.0.2 -> 10.0.0.1:9000 service=logging class=trusted", FG)]
    entries += [(x + tr, "EVENT 10.0.0.2 msg %d (blocked=False) → ACK" % (i + 1), FG) for i, x in enumerate(sends)]
    log_panel(d, t, entries)


@scene([
    "What if the controller crashes?",
    "The log server has a circuit breaker. It logs one warning, keeps serving clients without stalling, and retries every five seconds.",
    "A restarted controller has lost its rule table. So every two seconds, a reconcile loop compares the controller's rules with what the server expects, and re-installs anything missing.",
    "The network heals itself, with no operator involved.",
], tail=2.0)
def s_robust(d, t, c, D):
    title(d, t, "Scenario D · controller failure and recovery", "circuit breaker + reconcile loop")
    down = c[0] + 0.5 <= t < c[2] + 0.6
    topo(d, t, ctl_dim=down)
    if down:
        cross(d, CTL, RED, 40, P(t, c[0] + 0.5, 0.3), 9)
        text(d, (CTL[0], CTL[1] - 85), "Ctrl+C", 22, RED, P(t, c[0] + 0.5, 0.3), "bold")
    tr = 1.2
    if t >= c[1]:
        sends = [c[1] + 0.5 + 0.5 * k for k in range(40) if c[1] + 0.5 + 0.5 * k < c[2]]
        flow_packets(d, t, [H3, S1, H1], sends, tr, RED)
        # retries every "5 s" that fail at NAT
        for k in range(4):
            t0 = c[1] + 1.2 + k * 2.2
            if t0 < c[2]:
                f = (t - t0) / 0.9
                if 0 <= f <= 1:
                    p = path_point([H1, NAT, lerp(NAT, CTL, 0.5)], f)
                    dot(d, p, PURPLE, 9)
                elif 1 < f < 1.5:
                    cross(d, lerp(NAT, CTL, 0.5), RED, 14, 1 - (f - 1) * 2, 5)
        badge(d, (H1[0], H1[1] + 110), "serving · every reply ACK", BLUE, P(t, c[1], 0.4) * (1 - P(t, c[2], 0.4)), 20)
    # after restart: empty table, reconcile
    if t >= c[2]:
        empty = 1 - P(t, c[2] + 3.6, 0.3)
        badge(d, (CTL[0] - 330, CTL[1]), "rules: 0", MUTED, P(t, c[2] + 0.6, 0.3) * empty, 20)
        for k in range(3):
            t0 = c[2] + 1.2 + k * 2.0
            pulse(d, t, E_REST, t0, 1.0, PURPLE, 10, "GET /firewall/rules" if k == 0 else None)
        pulse(d, t, E_REST, c[2] + 2.4, 1.0, PURPLE, 10, None)
        pulse(d, t, [CTL, S1], c[2] + 3.3, 0.6, TEAL, 11, "FlowMod ×2")
        ra = P(t, c[2] + 3.9, 0.4)
        badge(d, (S1[0], S1[1] + 190), "priority=300,ip,nw_src=10.0.0.2 goto_table:1", GREEN, ra, 18)
        badge(d, (S1[0], S1[1] + 235), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, ra, 18)
        badge(d, (CTL[0] - 330, CTL[1]), "rules: 2", GREEN, ra, 20)
    if t >= c[3]:
        text(d, (720, 900), "✓ healed automatically", 34, GREEN, P(t, c[3], 0.5), "bold")
    entries = [(c[1], "WARNING: controller 10.0.0.254:8080 unreachable; continuing without blocking, retry in 5s", YELLOW),
               (c[1] + 1.5, "EVENT 10.0.0.3 msg 12 (blocked=False) → ACK", FG),
               (c[2] + 1.6, "controller 10.0.0.254:8080 reachable again", GREEN),
               (c[2] + 3.6, "ACTION: ALLOW rule for trusted source 10.0.0.2 installed on the controller", GREEN),
               (c[2] + 3.8, "ACTION: BLOCK rule for 10.0.0.3 installed on the controller", RED)]
    log_panel(d, t, entries)


@scene([
    "Detection doesn't have to come from the application alone.",
    "The controller's statistics poller turns per-flow byte counters into per-source throughput.",
    "With bandwidth detection enabled, a source measured above a threshold, say 3 megabits per second, is blocked on the switch's own evidence.",
], tail=2.5)
def s_bw(d, t, c, D):
    title(d, t, "Scenario E · detection from switch statistics", "log_server.py --bw-threshold-mbps 3")
    topo(d, t)
    for k in range(40):
        t0 = c[1] + 0.3 + k * 0.9
        if t0 < D:
            pulse(d, t, [CTL, S1], t0, 0.4, YELLOW, 7)
            pulse(d, t, [S1, CTL], t0 + 0.4, 0.4, YELLOW, 7)
    # chart
    a = P(t, c[1], 0.5)
    X0, Y0, X1, Y1 = 1360, 230, 1860, 700
    box(d, (X0 - 30, Y0 - 50, X1 + 20, Y1 + 80), GRID, a, fill=(18, 25, 40), r=14, w=2)
    text(d, (X0, Y0 - 22), "h3 throughput measured at s1 (Mbit/s)", 20, MUTED, a, anchor="lm", base=(18, 25, 40))
    ymax = 8.0
    ty = lambda v: Y1 - (v / ymax) * (Y1 - Y0)
    for v in (0, 2, 4, 6, 8):
        d.line((X0, ty(v), X1, ty(v)), fill=mix(GRID, a, (18, 25, 40)), width=1)
        text(d, (X0 - 8, ty(v)), str(v), 16, MUTED, a, anchor="rm", base=(18, 25, 40))
    thr = P(t, c[2], 0.5)
    d.line((X0, ty(3), X1, ty(3)), fill=mix(YELLOW, thr * a, (18, 25, 40)), width=3)
    text(d, (X1, ty(3) - 16), "threshold 3 Mbps", 18, YELLOW, thr * a, anchor="rm", base=(18, 25, 40))
    # illustrative series: ramps up, block, falls to 0
    t_start = c[1] + 0.5
    t_cross = c[2] + 3.0
    t_block = t_cross + 2.2

    def val(tt):
        if tt < t_start:
            return 0
        if tt < t_block:
            return min(7.2, 0.2 + (tt - t_start) * 6.5 / max(0.1, t_cross - t_start + 0.6))
        return max(0.0, 7.2 - (tt - t_block) * 9)
    span = D - t_start
    pts = []
    tt = t_start
    while tt <= t:
        pts.append((X0 + (tt - t_start) / span * (X1 - X0), ty(min(ymax, val(tt)))))
        tt += 1 / FPS
    if len(pts) > 1:
        d.line(pts, fill=mix(RED, a, (18, 25, 40)), width=4)
    text(d, (X0 + 250, Y1 + 40), "illustrative trace", 16, MUTED, a, base=(18, 25, 40))
    # flood packets
    sends = [c[1] + 0.3 + k * 0.18 for k in range(400) if c[1] + 0.3 + k * 0.18 < D]
    flow_packets(d, t, [H3, S1, H1], sends, 1.2, RED, r=8, drop_after=t_block)
    pulse(d, t, E_REST, t_cross + 0.4, 1.0, PURPLE, 11, "POST /firewall/block")
    pulse(d, t, [CTL, S1], t_cross + 1.45, 0.6, TEAL, 11, None)
    badge(d, (S1[0], S1[1] + 190), "priority=100,ip,nw_src=10.0.0.3 actions=drop", RED, P(t, t_block, 0.4), 18)
    tb = P(t, t_cross, 0.4)
    para(d, (X0 - 20, Y1 + 100), "ALERT: 10.0.0.3 measured 7.2 Mbps by the switch → BLOCKED", 20, 520, YELLOW, tb, "mono")


@scene([
    "To measure the benefit, an experiment harness compares three conditions: idle; a static baseline, where the log server only monitors; and the dynamic firewall.",
    "Each runs at low, medium, and high flood intensity, measuring the legitimate host's throughput, latency and loss, plus the attacker's accepted traffic, time to block, and packets dropped.",
    "The harness and analysis scripts are written and tested; running them on Mininet is the next step.",
    "Unit tests and a localhost end-to-end test cover the rule engine, the policy server, and real OpenFlow message serialisation.",
])
def s_eval(d, t, c, D):
    title(d, t, "Evaluation and testing", "experiments/run_experiments.py · experiments/analyze.py · tests/")
    a = P(t, c[0], 0.5)
    conds = [("idle", "no flood (reference)", MUTED), ("static", "flood, monitor-only (--no-block)", ORANGE), ("dynamic", "flood, SDN block active", GREEN)]
    ints = [("low", "1 conn · 20 msg/s"), ("medium", "1 conn · unthrottled"), ("high", "4 conns · saturates 10 Mbit/s")]
    GX, GY = 100, 230
    cw, ch = 280, 110
    for j, (n, s) in enumerate(ints):
        b = P(t, c[1] + j * 0.3, 0.4)
        text(d, (GX + 330 + j * cw + cw / 2, GY), n, 28, FG, b, "bold")
        text(d, (GX + 330 + j * cw + cw / 2, GY + 34), s, 18, MUTED, b)
    for i, (n, s, col) in enumerate(conds):
        b = P(t, c[0] + 0.6 + i * 0.6, 0.4)
        y = GY + 70 + i * ch
        text(d, (GX, y + ch / 2 - 14), n, 30, col, b, "bold", anchor="lm")
        text(d, (GX, y + ch / 2 + 20), s, 18, MUTED, b, anchor="lm")
        for j in range(3):
            bb = min(b, P(t, c[1] + j * 0.3, 0.4)) if i else b * P(t, c[1], 0.4) * (1 if j == 0 else 0.35)
            box(d, (GX + 340 + j * cw, y + 8, GX + 320 + (j + 1) * cw, y + ch - 8), col, bb, r=12, w=2)
            text(d, (GX + 330 + j * cw + cw / 2, y + ch / 2), "3 runs × 20 s", 20, MUTED, bb, base=PANEL)
    m = P(t, c[1] + 1.5, 0.5)
    text(d, (GX, 670), "Measured for h2 (legitimate)", 24, GREEN, m, "bold", anchor="lm")
    text(d, (GX, 710), "iperf throughput · ping RTT · loss", 22, FG, m, anchor="lm")
    text(d, (GX + 520, 670), "Measured for h3 (attacker)", 24, RED, m, "bold", anchor="lm")
    text(d, (GX + 520, 710), "accepted msgs · time to first block · switch drops", 22, FG, m, anchor="lm")
    s = P(t, c[2], 0.5)
    box(d, (GX, 760, GX + 1100, 840), YELLOW, s, r=12, w=2)
    text(d, (GX + 550, 800), "Status: harness + parsers tested on sample data · Mininet runs pending", 22, YELLOW, s, base=PANEL)
    # tests column
    tests = [("test_controller_logic.py", "rule engine, priorities, strict removal, real OFP serialisation"),
             ("test_policy_server.py", "classification, threshold, deny, controller-down, bandwidth"),
             ("test_experiment_parsing.py", "ping / iperf / client parsers"),
             ("local_test.py", "real Ryu + log server + clients on localhost")]
    for i, (n, s2) in enumerate(tests):
        b = P(t, c[3] + i * 0.5, 0.4)
        y = 260 + i * 140
        box(d, (1320, y, 1860, y + 120), BLUE, b)
        text(d, (1345, y + 35), "✓ " + n, 22, GREEN, b, "monob", anchor="lm", base=PANEL)
        para(d, (1345, y + 62), s2, 19, 490, MUTED, b, base=PANEL)


@scene([
    "To sum up: the application detects, the controller decides, and the switch enforces.",
    "Block for misbehaving hosts, deny for unauthorised flows, allow for trusted ones; all installed and removed at run time. That's a dynamic firewall.",
], tail=3.0)
def s_end(d, t, c, D):
    steps = [("Application", "detects", "TCP log server · sliding window · policy", BLUE),
             ("Controller", "decides", "Ryu · REST API · rule engine", PURPLE),
             ("Switch", "enforces", "OVS · OpenFlow 1.3 · table 0 drops", TEAL)]
    for i, (n, v, s, col) in enumerate(steps):
        a = P(t, c[0] + i * 1.1, 0.6)
        x = 330 + i * 630
        box(d, (x - 260, 230, x + 260, 520), col, a)
        text(d, (x, 300), n, 44, FG, a, "bold", base=PANEL)
        text(d, (x, 370), v, 38, col, a, "bold", base=PANEL)
        text(d, (x, 450), s, 21, MUTED, a, base=PANEL)
        if i < 2:
            b = P(t, c[0] + i * 1.1 + 0.6, 0.5)
            arrow(d, [(x + 270, 375), (x + 360, 375)], MUTED, b, 6, b)
    rules = [("BLOCK", "misbehaving hosts", RED), ("DENY", "unauthorised flows", ORANGE), ("ALLOW", "trusted sources", GREEN)]
    for i, (n, s, col) in enumerate(rules):
        a = P(t, c[1] + i * 0.8, 0.5)
        x = 330 + i * 630
        text(d, (x, 620), n, 40, col, a, "bold")
        text(d, (x, 670), s, 26, FG, a)
    e = P(t, c[1] + 4.5, 0.8)
    text(d, (W / 2, 790), "installed and removed at run time — no operator in the loop", 32, YELLOW, e, "bold")


# ================================================================ build
def synth(sentences_all, cache_dir):
    from kokoro_onnx import Kokoro
    k = None
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for i, s in enumerate(sentences_all):
        path = os.path.join(cache_dir, "%03d_%d.wav" % (i, int(__import__("hashlib").md5(s.encode()).hexdigest()[:8], 16)))
        if not os.path.exists(path):
            if k is None:
                k = Kokoro(os.path.join(HERE, "voice/kokoro.onnx"), os.path.join(HERE, "voice/voices.bin"))
            samples, sr = k.create(s, voice="am_michael", speed=1.0, lang="en-us")
            sf.write(path, samples, sr)
        data, sr = sf.read(path, dtype="float32")
        assert sr == SR
        out.append(data)
    return out


def main():
    out_path = sys.argv[1]
    preview = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--preview" else None
    os.environ.setdefault("PYTHONHASHSEED", "0")
    all_s = [s for (ss, _, _) in SCENES for s in ss]
    clips = synth(all_s, os.path.join(HERE, "tts_cache"))
    # timeline
    timeline, k, t_abs = [], 0, 0.0
    sentence_spans = []  # (abs start, abs end, text)
    for (ss, fn, tail) in SCENES:
        cues, t = [], LEAD
        for s in ss:
            dur = len(clips[k]) / SR
            cues.append(t)
            sentence_spans.append((t_abs + t, t_abs + t + dur, s, k))
            t += dur + GAP
            k += 1
        D = t - GAP + tail
        timeline.append((t_abs, D, fn, cues))
        t_abs += D
    total = t_abs
    print("total %.1fs, %d scenes" % (total, len(SCENES)))
    if preview is not None:
        s0, D, fn, cues = timeline[preview]
        for frac in (0.15, 0.4, 0.65, 0.9):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            fn(d, D * frac, cues, D)
            img.save(out_path.replace(".mp4", "_s%d_%d.png" % (preview, int(frac * 100))))
        return
    audio = np.zeros(int(total * SR) + SR, dtype=np.float32)
    for (st, en, s, idx) in sentence_spans:
        i0 = int(st * SR)
        audio[i0:i0 + len(clips[idx])] += clips[idx]
    wav = out_path + ".wav"
    sf.write(wav, audio, SR)
    # subtitles (SRT) too
    def ts(x):
        h, r = divmod(x, 3600); m, s = divmod(r, 60)
        return "%02d:%02d:%06.3f" % (h, m, s)
    with open(out_path.replace(".mp4", ".srt"), "w") as f:
        for i, (st, en, s, _) in enumerate(sentence_spans):
            f.write("%d\n%s --> %s\n%s\n\n" % (i + 1, ts(st).replace(".", ","), ts(en).replace(".", ","), s))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-i", wav,
                           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out_path],
                          stdin=subprocess.PIPE)
    nframes = int(total * FPS)
    si = 0
    for fi in range(nframes):
        ta = fi / FPS
        while si < len(timeline) - 1 and ta >= timeline[si][0] + timeline[si][1]:
            si += 1
        s0, D, fn, cues = timeline[si]
        t = ta - s0
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        fn(d, t, cues, D)
        # scene cross-fade from/to black
        cap = None
        for (st, en, s, _) in sentence_spans:
            if st - 0.1 <= ta <= en + GAP * 0.9:
                cap = s
        caption(d, cap, 1.0)
        fade = min(clamp(t / 0.35), clamp((D - t) / 0.35))
        if fade < 1:
            img = Image.blend(Image.new("RGB", (W, H), BG), img, fade)
        ff.stdin.write(img.tobytes())
        if fi % 600 == 0:
            print("frame %d / %d" % (fi, nframes), flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
