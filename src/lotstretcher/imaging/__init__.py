"""
Image-processing pipeline for downloaded vehicle photos, kept separate from
the scraping logic in lotstretcher.py so stages can be developed/tested in isolation
and new ones added without touching the scraper.

Modules:
  dedupe.py   -- perceptual-hash filtering of known junk/marketing template
                 images (e.g. a dealer's reused "check our reviews" card).
  classify.py -- CLIP zero-shot scene classification (exterior / interior /
                 marketing graphic / document).
  cutout.py   -- rembg background removal, with a quality gate that flags
                 unreliable segmentations (typical for tight detail shots
                 like a wheel or badge close-up) so callers can fall back
                 to the original photo instead of shipping a bad cutout.
  pipeline.py -- combines the above: trusts CLIP's label directly except
                 when it says "marketing," in which case rembg's foreground
                 margin is used as a tiebreaker (handles Tomball Ford's
                 photos that have dealer-branding baked into the pixels,
                 which can dominate CLIP's holistic scene score on an
                 otherwise-real exterior shot).

lotstretcher.py wires dedupe.py + pipeline.py + classify.py + cutout.py together in
download_photos() to sort each vehicle's gallery into images/exterior/ and
images/interior/, with white-background cutouts saved alongside originals
in images/exterior/cutout/ wherever the quality gate allows.
"""
