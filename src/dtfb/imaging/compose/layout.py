"""
Where each car cutout goes within the border's window.

A layout is a function (window, n_extra) -> [(box, anchor), ...], always
hero first. compose_hero() zips this 1:1 against the car_paths it's given,
so the caller (imaging/select.py, or a CLI) is responsible for handing
cars over in the same order the chosen layout expects.
"""
from __future__ import annotations

from PIL import Image


def compute_placement(car: Image.Image, box: tuple[int, int, int, int],
                       margin_frac: float = 0.06, anchor: str = "bottom") -> tuple[int, int, Image.Image]:
    """Scale `car` (already cropped to its visible pixels) to fit within
    `box` minus a margin. Returns (x, y, resized_car) without touching any
    canvas -- split out from pasting so the placement can be known ahead of
    the dim/vignette/glow passes, which need to know where the car will sit
    before it's actually drawn."""
    bl, bt, br, bb = box
    avail_w = (br - bl) * (1 - 2 * margin_frac)
    avail_h = (bb - bt) * (1 - 2 * margin_frac)

    scale = min(avail_w / car.width, avail_h / car.height)
    new_size = (max(1, round(car.width * scale)), max(1, round(car.height * scale)))
    # reducing_gap=2.0: cheap box pre-reduction before the LANCZOS pass,
    # same tradeoff as hero_video.py's per-frame resize -- see its comment.
    car_resized = car.resize(new_size, Image.LANCZOS, reducing_gap=2.0)

    x = bl + ((br - bl) - new_size[0]) // 2
    if anchor == "bottom":
        y = bb - int((bb - bt) * margin_frac) - new_size[1]
    else:
        y = bt + ((bb - bt) - new_size[1]) // 2

    return x, y, car_resized


def single_layout(window: tuple[int, int, int, int], n_extra: int = 0) -> list[tuple[tuple, str]]:
    """Just one car, filling the window. Centered rather than bottom-anchored
    -- bottom-anchoring reads right for the quad layout's hero (standing on
    "the ground" at the base of the frame, other cars above it), but for a
    solo shot with nothing else in the window it just leaves a lot of dead
    space up top; centered uses the window evenly. n_extra is ignored (kept
    for a consistent layout-function signature)."""
    return [(window, "center")]


def corners_layout(window: tuple[int, int, int, int], n_extra: int) -> list[tuple[tuple, str]]:
    """Hero fills the window (bottom-anchored) below the two corner
    accents; up to 2 smaller accent shots tucked into the top corners.

    Corner placement (rather than e.g. side-by-side thirds) keeps the hero
    the clear focal point at full size instead of shrinking everything to
    fit -- accents read as supporting angles, not equal billing.

    Hero's box is bounded to start below the corner accents (not the full
    window) -- see quad_layout()'s docstring for why: a hero that isn't
    always the same wide/low 3/4 shot (e.g. hero_video.py's carousel,
    which cycles through every angle) can be tall enough, scaled to fill
    the window, to visually paint over the accents above it if its box
    isn't actually bounded away from theirs.
    """
    wl, wt, wr, wb = window
    ww, wh = wr - wl, wb - wt

    accent_w, accent_h = round(ww * 0.40), round(wh * 0.30)
    corners = [(wl, wt), (wr - accent_w, wt)]
    accent_boxes = [((ax, ay, ax + accent_w, ay + accent_h), "center") for ax, ay in corners[:n_extra]]

    hero_top = wt + accent_h + round(wh * 0.02) if accent_boxes else wt
    boxes = [((wl, hero_top, wr, wb), "bottom")]
    boxes.extend(accent_boxes)
    return boxes


def quad_layout(window: tuple[int, int, int, int], n_extra: int = 3) -> list[tuple[tuple, str]]:
    """Hero at the bottom + 3 fixed accent slots: front in the top-left
    corner, rear/back in the top-right corner, and a wide strip for a full
    side profile filling the gap between them. Always 4 slots regardless
    of n_extra -- pair with imaging/select.py::pick_for_quad_layout() to
    hand compose_hero() the [hero, front, back, side] car order this
    expects.

    Hero's box stops just below the accent row instead of spanning the
    full window. compose_placement() only guarantees a car fits inside
    its OWN box -- it doesn't know or care about any other box's content,
    and compose_hero() paints the hero last (on top), so a hero box that
    overlaps the accent boxes lets a sufficiently tall hero visually cover
    them. That's invisible for a still image (the hero is always the same
    wide/low 3/4 shot, which is naturally short enough not to reach the
    accent row even when the box was the full window) -- but hero_video.py
    cycles the hero through every angle, including squarer ones (a
    straight rear shot, a wheel money shot) that DO reach that high once
    scaled to fill a same-sized box. Bounding the hero box itself is the
    fix that works regardless of which shot ends up in it, rather than
    special-casing "unless it's this angle."
    """
    wl, wt, wr, wb = window
    ww, wh = wr - wl, wb - wt

    gap = round(ww * 0.02)  # breathing room between the 3 top boxes -- these
    # used to be exactly edge-to-edge (zero gap), which let normal per-box
    # margins and any glow effect visibly touch/merge between neighbors.

    corner_w, corner_h = round(ww * 0.32), round(wh * 0.26)
    top_left = (wl, wt, wl + corner_w, wt + corner_h)
    top_right = (wr - corner_w, wt, wr, wt + corner_h)

    side_h = round(corner_h * 0.85)
    side_box = (top_left[2] + gap, wt, top_right[0] - gap, wt + side_h)

    accent_row_bottom = max(top_left[3], top_right[3], side_box[3])
    hero_box = (wl, accent_row_bottom + gap, wr, wb)

    return [
        (hero_box, "bottom"),
        (top_left, "center"),
        (top_right, "center"),
        (side_box, "center"),
    ]


# Target box aspects, all measured against the fleet's cutouts: 215 real
# exterior cutouts run 1.19:1 to 3.23:1 with a median of 1.80:1. Vehicles
# are wide objects, and a box that isn't roughly that shape wastes its
# own area no matter how big it is.
CONVEYOR_MAX_HERO_ASPECT = 2.1   # cap for the side-by-side hero on a wide frame
CONVEYOR_HERO_ASPECT = 1.55      # stacked hero, matching the square layout's natural box
CONVEYOR_ACCENT_ASPECT = 1.75    # stacked accents, just under the cutout median


def conveyor_layout(window: tuple[int, int, int, int], n_extra: int = 2) -> list[tuple[tuple, str]]:
    """Hero + 2 top-corner accents for hero_video.py's 3-slot conveyor.

    Same idea as corners_layout, retuned for a frame where all three
    slots are ALWAYS occupied -- corners_layout has to look right with 0,
    1, or 2 accents and with a hero that may be the only thing on screen,
    so it leaves generous air. Here the video was measurably empty:
    832px of content in a 1046px window (20% dead), with the worst gap
    236px of bare background between a short accent and the hero.

    Two changes, each aimed at one measured cause:
      - Boxes are bigger (0.46w x 0.34h vs 0.40 x 0.30) and the row gap
        tighter, so the accents themselves occupy more of the top band.
      - The hero box takes everything below that band; hero_video.py
        pairs this with a much smaller inset than the still pipeline's
        (see HERO_MARGIN_FRAC), since a video frame is viewed briefly and
        wants to read big, where a still is looked at.

    Accents stay CENTER-anchored. Bottom-anchoring them was tried, and it
    does close the gap to the hero -- a wide side profile only fills
    174px of a 356px box, so dropping it to a shared baseline with the
    other accent cut the worst gap from 236px to 73px. But it reads
    wrong: a short wide shot pinned to the bottom of an invisible box
    looks like it's sinking, with all of its slack stacked above it,
    while centering splits the slack evenly and reads as deliberate
    letterboxing. Note this only affects wide/short shots -- an accent
    that fills its box height (a wheel, a straight-on front) lands within
    ~1px either way, so this is not a general "align everything" knob.

    n_extra is accepted for signature compatibility and ignored -- the
    conveyor is always exactly hero + 2.
    """
    wl, wt, wr, wb = window
    ww, wh = wr - wl, wb - wt

    accent_w, accent_h = round(ww * 0.46), round(wh * 0.34)
    gap = round(wh * 0.015)

    left_box = (wl, wt, wl + accent_w, wt + accent_h)
    right_box = (wr - accent_w, wt, wr, wt + accent_h)

    # Spanning the full window width is right on a square frame and wrong
    # on a wide one. Measured against the fleet's cutouts (median 1.80:1):
    # a square window gives a 1.55:1 hero box that a median vehicle fills
    # 86% of, but a 16:9 window gives 2.76:1 and fill drops to 65% -- the
    # vehicle fits to height and leaves air down both sides. Capping the
    # box aspect and centring it restores 86%. The cap is above a square
    # window's natural 1.55, so this changes nothing there.
    hero_top = wt + accent_h + gap
    hero_h = wb - hero_top
    hero_w = min(ww, round(hero_h * CONVEYOR_MAX_HERO_ASPECT))
    hero_l = wl + (ww - hero_w) // 2
    hero_box = (hero_l, hero_top, hero_l + hero_w, wb)

    return [(hero_box, "bottom"), (left_box, "center"), (right_box, "center")]


def conveyor_stack_layout(window: tuple[int, int, int, int], n_extra: int = 2) -> list[tuple[tuple, str]]:
    """The same 3-slot conveyor, stacked instead of side-by-side, for a
    window taller than it is wide.

    conveyor_layout cannot be reused on a vertical frame. Measured on a
    1080x1920 canvas it produces a 0.87:1 hero box and 0.76:1 accent
    boxes -- portrait boxes, which is the worst possible shape for an
    object whose median aspect is 1.80:1. A median vehicle fills 48% of
    that hero box and 42% of an accent. The frame isn't badly composed,
    it's mostly empty.

    So the accents move above and below the hero rather than sitting in
    its top corners. The important detail is that they must also be
    NARROWER than the hero: everything in a tall frame is width-
    constrained, so three full-width slots would draw the vehicle at
    exactly the same size in all three and the hero/accent hierarchy
    would vanish entirely. 0.62 width keeps roughly the same
    accent-to-hero size ratio the square layout has (0.46/1.0).

    Boxes are sized to the content's aspect rather than to equal shares
    of the height, and whatever height is left over becomes air between
    them -- a tall composition wants that spacing, and sizing the boxes
    to fill the frame is what produced the 48% number above.

    Slot order matches conveyor_layout's [hero, outgoing, incoming], so
    hero_video.py's animation is unchanged: a shot fades in at the BOTTOM
    accent, grows into the hero, shrinks into the TOP accent and fades
    out. The travel reads upward, which suits a vertical frame.
    """
    wl, wt, wr, wb = window
    ww, wh = wr - wl, wb - wt

    hero_w = ww
    hero_h = min(wh, round(hero_w / CONVEYOR_HERO_ASPECT))
    accent_w = round(ww * 0.62)
    accent_h = round(accent_w / CONVEYOR_ACCENT_ASPECT)

    hero_top = wt + (wh - hero_h) // 2
    hero_box = (wl, hero_top, wr, hero_top + hero_h)

    accent_l = wl + (ww - accent_w) // 2
    # Each accent is centred in the band left over above/below the hero,
    # so the leftover height reads as deliberate spacing rather than as
    # slack pinned to one edge.
    top_band = hero_top - wt
    bottom_band = wb - (hero_top + hero_h)
    top_y = wt + max(0, (top_band - accent_h) // 2)
    bottom_y = hero_top + hero_h + max(0, (bottom_band - accent_h) // 2)

    top_box = (accent_l, top_y, accent_l + accent_w, top_y + accent_h)
    bottom_box = (accent_l, bottom_y, accent_l + accent_w, bottom_y + accent_h)

    return [(hero_box, "center"), (top_box, "center"), (bottom_box, "center")]


def conveyor_for_window(window: tuple[int, int, int, int]):
    """Whichever conveyor arrangement suits this frame's shape. Stacked
    once the window is taller than it is wide; side-by-side otherwise, so
    square and 16:9 both keep the tuned original."""
    wl, wt, wr, wb = window
    return conveyor_stack_layout if (wr - wl) < (wb - wt) else conveyor_layout


LAYOUTS = {
    "single": single_layout,
    "corners": corners_layout,
    "quad": quad_layout,
    "conveyor": conveyor_layout,
    "conveyor_stack": conveyor_stack_layout,
}
