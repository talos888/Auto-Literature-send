# Weekly Literature Radar

This is the first part of the automation: weekly discovery and classification
of new arXiv papers related to AI-driven or automated scientific laboratories.
It sends the weekly report to the address configured in the `MAIL_TO` GitHub
Actions secret.

## What It Does

- Runs every Monday at 08:00 Asia/Singapore through GitHub Actions.
- Searches arXiv over the previous 14 days, then skips papers already seen in
  earlier runs.
- Scores candidates with transparent keyword rules.
- Optionally asks DeepSeek to classify likely matches when `DEEPSEEK_API_KEY` is set
  as a GitHub Actions secret.
- Emails the Markdown report and uploads Markdown/JSON output as a workflow artifact.

## Files

- `config.json`: topic keywords, arXiv categories, and LLM settings.
- `run_literature_radar.py`: dependency-free Python scanner and classifier.
- `send_email.py`: dependency-free SMTP sender for the generated Markdown report.
- `tests/`: standard-library unit tests for scoring and secret handling.
- `.github/workflows/weekly-literature-radar.yml`: scheduled GitHub Actions job.

## GitHub Setup

1. Put these files in a GitHub repository.
2. Optional but recommended: add `DEEPSEEK_API_KEY` under repository `Settings -> Secrets and variables -> Actions`.
3. Add SMTP email secrets under the same page:
   - `SMTP_HOST`
   - `SMTP_PORT`
   - `SMTP_USERNAME`
   - `SMTP_PASSWORD`
   - `SMTP_SSL` (`true` for implicit SSL, usually port 465; `false` or empty for STARTTLS, usually port 587)
   - `MAIL_FROM`
   - `MAIL_TO`
4. Open the `Actions` tab and run `Weekly Literature Radar` manually once.
5. Check the `literature-radar-report` artifact.

Without `DEEPSEEK_API_KEY`, the job still runs and uses rule-based filtering only.
Without the SMTP secrets, the email step fails because there is no sender account.
`MAIL_TO` should be set to the receiving address, for example `you@example.com`.

## Secret Handling

- Never write API keys, SMTP passwords, or mail tokens into `config.json`, workflow YAML, README files, or command
  line arguments.
- Store the DeepSeek key and SMTP credentials only as GitHub Actions repository
  secrets.
- The workflow has read-only repository permissions and runs a secret hygiene
  scan before the literature radar starts.
- The script refuses secret-like fields such as `api_key`, `token`, or `password`
  in JSON config files.
- The email sender refuses credential-looking command-line arguments.
- Local `.env`, `*.key`, `*.pem`, and generated output folders are ignored by
  `.gitignore`.

## Manual Local Test

After generating a report, you can verify email configuration without sending:

```bash
python literature-radar/send_email.py --report literature-radar/out/literature-radar-YYYY-MM-DD.md --dry-run
```

Run the dependency-free unit tests:

```bash
python -m unittest discover -s literature-radar/tests
```
