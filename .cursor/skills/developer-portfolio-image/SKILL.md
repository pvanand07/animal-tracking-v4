---
name: developer-portfolio-image
description: Generate a polished 16:9 developer-portfolio infographic image (product + engineering showcase) for an app or project, suitable for a portfolio, LinkedIn project post, engineering case study, or architecture deck. Use when the user asks to create a project/product showcase image, portfolio image, one-pager, or "infographic" for something they built.
---

# Developer Portfolio Image

Produce a single 16:9 image that communicates, in under ~10 seconds: what the
product does, how it works, and what it's built with.

## Workflow

1. **Collect the required data.** Read [data-checklist.md](data-checklist.md) and ask
   the user for as many of those fields as possible (product name, value prop, key
   features, architecture/workflow, tech stack, screenshots, real metrics, etc.).
   Never invent data — especially metrics — that the user didn't supply. If a field is
   missing, drop that element from the image rather than fabricating it.

2. **Fill the generation prompt.** Read [prompt-template.md](prompt-template.md) and
   substitute the collected data into its `APPLICATION DATA` section. That file also
   contains the full visual-style rules, image structure (header, sources/inputs, core
   processing, product experience, tech stack bar), content-simplification rules, and a
   list of things to avoid — follow all of it.

3. **Generate the image** (image generation model, or an HTML/SVG artifact rendered to
   an image if no image model is available) using the filled-in prompt.

4. **Sanity-check against the "AVOID" list** in prompt-template.md before delivering —
   no fake metrics, no dense diagrams, no long paragraphs, no distorted mockups.

## Notes

- Adapt the "Core Processing / Architecture" pipeline wording to the product type
  (e.g. `Upload → Parse → Extract → Review` for document intelligence, `Catalog →
  Search → Recommend → Checkout` for e-commerce) — examples are in
  prompt-template.md.
- The product-experience mockup should be the largest section and should reconstruct a
  simplified UI rather than shrink a full screenshot into unreadable text.
