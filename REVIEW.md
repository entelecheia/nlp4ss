# Review instructions

Read by the required pre-merge reviewer (Claude Code, Codex, or Kimi Code) and human reviewers. This repository is mainly a course book (nlp4ss.jeju.ai); the Python package is small.

## Passes

- Bugs: notebooks that no longer run (broken setup cells, missing imports or install lines, wrong paths), broken book build configuration.
- Security: secrets or tokens in notebooks or their outputs, unsafe install sources.
- Compliance: the change matches its issue and keeps the scope it states.

## Repo focus

- Notebooks stay valid nbformat JSON. Edits keep outputs and metadata of untouched cells.
- Setup cells run on Colab and locally: Colab-only calls (`google.colab`, Drive mount) are guarded by a Colab check.
- The book is built without executing notebooks; a PR that changes build or deploy settings states how it was checked.

## What Important means here

Findings that stop a notebook or the book build from working, or expose a secret. Wording and style are nits.

## Cap the nits

At most 5 nits per review; summarize the rest as a count.

## Do not report

- Notebook output cells and embedded images (base64), unless they contain secrets.
- Generated files: lock files (style only), `CHANGELOG.md`, `_version.py`, built book output.
