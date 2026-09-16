# AGENTS.md

Notes for agents (and humans) working on agentknit.

## Tests must never read the developer's machine

**A test that passes in CI and fails locally — or the reverse — is a bug in the
test, never an acceptable state.** It must be fixed, not explained away as
"pre-existing" or "unrelated".

agentknit reads user-level config at runtime: `~/.agentknit/hooks.json`,
`~/.claude/CLAUDE.md`, `~/.local/share/...` log and history dirs. Any test that
reaches those files inherits whatever the developer happens to have configured.

Guards in place:

* `conftest.py` sets `$HOME` to an empty per-run directory for **every** test
  (autouse). `Path.home()` and `os.path.expanduser` both honour `$HOME`, so
  this neutralises all user-level layers at once.
* CI plants a hostile `~/.agentknit/hooks.json` and `~/.claude/CLAUDE.md`
  before running pytest, so a leak fails the build rather than hiding behind
  CI's bare home.

When adding a feature that reads a new path under `$HOME`, or any other
machine-specific state (git config, env vars, the clock, the network), isolate
it in the test rather than asserting on whatever the machine returns.

## Conventions

* Commit author: `Martin Monperrus (AI-assisted) <martin.monperrus+ai@gnieh.org>`.
* Never break backward compatibility.
* Tests live at the repo root as `test_*.py`; run `python -m pytest -q`.
* `ruff check agentknit/` and `python -m mypy agentknit/` must stay clean.
