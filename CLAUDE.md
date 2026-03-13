# CLAUDE.md — Project Rules for Claude Code

## Sensitive Information Protection (MANDATORY)

These rules apply to EVERY commit and push operation, no exceptions.

1. **Pre-commit secret scan**: Before executing any `git commit` or `git push`, scan ALL staged/changed files for hardcoded API keys, passwords, tokens, secrets, or other sensitive information. Look for patterns like `sk-`, `sk-or-`, `ghp_`, `Bearer`, `password=`, `secret=`, `token=`, and any string that looks like a credential.

2. **Block on detection**: If any sensitive information is found, STOP immediately. Do NOT commit or push. Report the exact file path and line number, then refactor the code to read from environment variables (e.g., `os.environ["VAR_NAME"]`) and store the actual values in a `.env` file.

3. **Protect .env**: Ensure `.env` is listed in `.gitignore` before any commit. It must NEVER be committed to the repository.

4. **Gate on clean scan**: Only proceed with commit/push after confirming zero hardcoded secrets in all files being committed.

5. **Proactive interception**: Even if the user says "push" or "commit" without mentioning a scan, always perform the secret scan first. This rule cannot be skipped.
