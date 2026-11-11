# Claude API credentials and safety

## Setup (one time, Mac)

**Option A, preferred: OAuth profile, no static key**
```bash
# install the `ant` CLI per the Claude platform docs, then:
ant auth login --profile chiprl     # browser login, pick the chiprl workspace
export ANTHROPIC_PROFILE=chiprl     # put this in .envrc
ant auth status                     # confirms which credential wins
```
Tokens are short-lived and refresh automatically. Nothing secret sits in the repo or your shell
history. The SDK's zero-arg `Anthropic()` (used by `chiprl.llm.auth.make_client`) picks it up.

**Option B: API key in the macOS Keychain**
1. In the Console, create a workspace named `chiprl`, set a **monthly spend limit** (start at $50),
   and create a key in that workspace only. Never use a key from your default workspace.
2. Store it in the Keychain (prompts for the value, so it never hits shell history):
   ```bash
   security add-generic-password -a "$USER" -s chiprl-anthropic -w
   ```
3. Load it only into the shell that needs it. `.envrc` (direnv) can do this:
   ```bash
   export ANTHROPIC_API_KEY="$(security find-generic-password -a "$USER" -s chiprl-anthropic -w)"
   ```
4. Rotate it from the Console if it ever appears in a log, a paste, or a commit.

Never put a key in `.env`, `config*.json`, notebooks, `configs/`, or a CLI argument (`ps` shows
arguments). `make_client()` refuses an explicit key argument by design.

## Guards in the repo
| Risk | Guard |
|---|---|
| Key committed | `.gitignore` covers `.env*`, `runs/`; gitleaks pre-commit hook; GitHub secret scanning + push protection (turn on in repo settings) |
| Key in logs or trajectories | `auth.redact()` on every persisted string; thinking signatures dropped |
| Stale env var outranks profile | `auth.credential_source()` warns |
| Runaway spend | per-run `--max-usd` hard cap (`costs.Budget`, raises and cancels pending work) + Console workspace limit |
| Model-generated RTL executing on your Mac | runs only inside tool wrappers: rlimits, temp dirs, no network for Docker tools; static policy bans system tasks |
| Reward hacking / grader leakage | hidden tb and reference never returned by tools; `simulate` rate-limited; reward not shown on submit; auditor flags passes |
| CI spending money / CI key theft | no stored key at all: GitHub OIDC -> Claude **Workload Identity Federation** short-lived token. `llm-eval.yml` is manual-only in a protected `llm-evals` environment; `claude.yml` answers only the repo OWNER and reviews only same-repo PRs |
| Prompt injection via tool output | tool results are compact JSON with truncated diagnostics; specs come only from the task set |

## CI auth: Workload Identity Federation (no secret in GitHub)
1. Console (chiprl workspace): add GitHub as a federation issuer
   (`https://token.actions.githubusercontent.com`), create a service account, and a federation rule
   matching `repo:onyewuenyi/chiprl:*` (narrow to `environment:llm-evals` for the eval job).
2. Repo Settings -> Variables: `ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID`,
   `ANTHROPIC_SERVICE_ACCOUNT_ID`. These are identifiers, not secrets.
3. Environments -> `llm-evals`: required reviewer = you.
4. Turn on secret scanning + push protection for the repo.
