"""
Expand an inventory/listing URL -- a filtered search-results page, e.g.
.../inventory/all-vehicles/models-Dodge/models-Kia/models-GMC/ or
.../inventory/all-vehicles/text-raptor/ -- into the individual vehicle
detail page (VDP) URLs it contains, following pagination automatically.

Lets a user filter on the dealer site itself (any combination of make/
model/price/etc the site's own UI supports) and just hand lotstretcher the
resulting listing URL instead of copying out each VDP URL by hand.
"""
from __future__ import annotations

import re
import sys
import time
from urllib.parse import urljoin, urlparse

from lotstretcher.scrape import USER_AGENT, fetch_rendered_html

VDP_LINK_RE = re.compile(r'/vehicle/[A-HJ-NPR-Z0-9]{17}/[^"\'<>]*')
# [\d,]+ not \d+ -- confirmed real bug this fixes: the site renders this as
# "1,352 Vehicles Found" for anything over 999, and \d+ alone silently
# matched only the digits after the last comma ("352"), truncating every
# 4+-digit total to its last 3 digits without ever raising an error. That
# wrong (too-small) expected_total then made every downstream "did we get
# everything" check -- including the stall-retry logic below -- trust a
# crawl that stopped hundreds of vehicles short of the real inventory.
VEHICLE_COUNT_RE = re.compile(r'([\d,]+)\s+vehicles?\s+found', re.IGNORECASE)

# A safety cap, not an expected real value -- just a backstop against an
# infinite loop if the page-count/new-results termination logic ever fails
# to trigger on a differently-structured listing page. Confirmed this needs
# real headroom, not a token one: this dealer's own combined "all-vehicles"
# listing genuinely runs to ~1,500 vehicles (~125+ pages at 12/page) once
# VEHICLE_COUNT_RE's comma-parsing bug was fixed -- the old MAX_PAGES=50
# would have silently truncated a real crawl at less than half that.
MAX_PAGES = 300

# A single page fetch failure is retried this many extra times (with a
# short pause) before giving up on it -- confirmed real cause worth
# retrying for: this machine's wired NIC throws brief net::ERR_NETWORK_CHANGED
# blips (visible as RX drops on the interface) that a few seconds' wait
# reliably clears, and re-fetching one page is far cheaper than losing the
# rest of the crawl to what turns out to be a few seconds of flakiness.
PAGE_RETRIES = 2

# How close to the listing's own reported total a zero-new-vehicles page
# has to leave us before we trust it as a genuine "that's everything"
# rather than retrying as a possibly-stale page.
STALL_TOLERANCE = 0.9


def _count_new_links(html: str, seen_vins: set[str], urls: list[str], listing_url: str) -> int:
    """Extracts VDP links from `html`, appends any not already in
    `seen_vins` to `urls` (mutating both), and returns how many were new."""
    new_on_this_page = 0
    for path in VDP_LINK_RE.findall(html):
        vin = path.split("/")[2]
        if vin not in seen_vins:
            seen_vins.add(vin)
            urls.append(urljoin(listing_url, path))
            new_on_this_page += 1
    return new_on_this_page


def is_vdp_url(url: str) -> bool:
    """A vehicle detail page URL has /vehicle/<VIN>/ in its path; anything
    else (a filtered inventory/listing/search-results page) needs
    expanding via expand_listing_url() first."""
    return "/vehicle/" in urlparse(url).path


def _fetch_listing_page(browser, url: str) -> str:
    """Fresh CONTEXT per page fetch (fresh cookies/storage, same reasoning
    as scrape.py's VDP fetcher: Cloudflare's bot score seems to accumulate
    across navigations in one session) -- but the browser PROCESS itself is
    shared across the whole crawl, not relaunched per page. Confirmed real
    cost of the old per-page-browser approach: a full Chromium launch is
    ~1-2s of pure overhead, which is negligible for one page but adds
    minutes of pure waiting across a 125+ page crawl of this dealer's real
    ~1,500-vehicle combined inventory, for zero benefit -- a fresh context
    is already enough isolation, a fresh process bought nothing extra."""
    ctx = browser.new_context(user_agent=USER_AGENT, viewport={"width": 1366, "height": 900})
    page = ctx.new_page()
    try:
        return fetch_rendered_html(page, url)
    finally:
        ctx.close()


def expand_listing_url(playwright, listing_url: str, headed: bool = False) -> tuple[list[str], bool, int | None]:
    """Crawls a filtered inventory listing page and returns every vehicle's
    full VDP URL it links to, in the order first seen, de-duplicated by VIN.

    Pagination is dealer.com's own "srp-page-N/" URL segment appended to
    the listing URL (confirmed live: a 14-result filtered listing showed
    12 vehicles on the base URL, the remaining 2 at .../srp-page-2/) --
    simple enough to construct directly rather than needing to drive any
    "load more" UI interaction. Stops when a page adds no new vehicles, or
    once the listing's own reported "N vehicles found" count is reached,
    whichever comes first.

    A page fetch failing partway through (network blip, a harder
    Cloudflare challenge on a later page) stops pagination and returns
    whatever was already collected, rather than losing it -- confirmed
    real case: a 10+ page "models-Ford" listing hit a transient network
    error on page 10, and letting that exception propagate out of this
    function (as it used to) would have discarded the ~100+ vehicles
    already found on pages 1-9 along with it. Partial results are still
    useful; the caller sees a short list rather than nothing.

    Returns (urls, complete, expected_total) -- `complete` is False whenever a page fetch
    errored out early, True when pagination stopped because a page
    repeated (or the listing's own count was reached). Confirmed real
    incident this distinction exists to prevent: a network blip on page 8
    of the ~150-vehicle "all-vehicles" listing silently truncated a crawl
    to 83 URLs, and --sync's delist step (which has no other way to know
    the crawl was incomplete) then flagged 20 real, still-listed vehicles
    as delisted because they simply weren't on the pages that got fetched.
    Callers doing anything as consequential as "diff against everything
    previously scraped" MUST check `complete` first.

    `expected_total` is the listing's own "N vehicles found" count (None if
    that text couldn't be parsed off the page) -- exposed so a caller can
    print an explicit "crawled X of Y the site itself claims to have"
    confirmation before starting any processing, rather than that check
    happening invisibly inside this function.
    """
    base = listing_url.rstrip("/") + "/"
    seen_vins: set[str] = set()
    urls: list[str] = []

    browser = playwright.chromium.launch(
        headless=not headed,
        args=["--disable-blink-features=AutomationControlled"],
    )
    try:
        return _crawl_pages(browser, base, seen_vins, urls)
    finally:
        browser.close()


def _crawl_pages(browser, base: str, seen_vins: set[str], urls: list[str]) -> tuple[list[str], bool, int | None]:
    expected_total: int | None = None
    complete = True

    for page_num in range(1, MAX_PAGES + 1):
        page_url = base if page_num == 1 else f"{base}srp-page-{page_num}/"
        if page_num > 1:
            time.sleep(2)  # same Cloudflare-friendly pacing as the main vehicle loop
        html = None
        for attempt in range(PAGE_RETRIES + 1):
            try:
                html = _fetch_listing_page(browser, page_url)
                break
            except Exception as e:
                last_error = e
                if attempt < PAGE_RETRIES:
                    print(f"    ! listing page {page_num} failed ({e}); retrying...", file=sys.stderr)
                    time.sleep(3)
        if html is None:
            print(f"    ! listing page {page_num} failed after {PAGE_RETRIES + 1} attempt(s) "
                  f"({last_error}); returning {len(urls)} vehicle(s) found so far (PARTIAL)", file=sys.stderr)
            complete = False
            break

        if expected_total is None:
            m = VEHICLE_COUNT_RE.search(html)
            if m:
                expected_total = int(m.group(1).replace(",", ""))

        new_on_this_page = _count_new_links(html, seen_vins, urls, base)

        if new_on_this_page == 0:
            # Confirmed real case this guards against: a real sync run
            # stopped here at page ~14 (168 vehicles) with no fetch error
            # at all, while the listing's own count said 502 -- almost
            # certainly a stale/cached page served back under load (this
            # site's rate limiting or Cloudflare), not an actual last page.
            # A genuine last page keeps returning zero new vehicles on a
            # retry; a stale one usually doesn't. Only trust a single
            # zero-new page as "done" outright when we don't have an
            # expected_total to check it against, or we're already close
            # to it -- otherwise retry, and if it's still short after
            # retrying, call the crawl incomplete rather than silently
            # accepting a short list as the whole inventory.
            close_to_total = expected_total is None or len(seen_vins) >= expected_total * STALL_TOLERANCE
            if close_to_total:
                break
            stalled = True
            for attempt in range(PAGE_RETRIES):
                print(f"    ! listing page {page_num} returned no new vehicles but only "
                      f"{len(seen_vins)}/{expected_total} found so far -- retrying (stale page?)...",
                      file=sys.stderr)
                time.sleep(3)
                try:
                    html = _fetch_listing_page(browser, page_url)
                except Exception:
                    continue
                if _count_new_links(html, seen_vins, urls, base) > 0:
                    stalled = False
                    break
            if stalled:
                print(f"    ! listing page {page_num} still returned no new vehicles after retrying, "
                      f"but only {len(seen_vins)}/{expected_total} found -- treating crawl as incomplete "
                      "rather than trusting a possibly-stale page", file=sys.stderr)
                complete = False
                break
            continue
        if expected_total is not None and len(seen_vins) >= expected_total:
            break  # got everything the listing itself claims to have

    return urls, complete, expected_total
