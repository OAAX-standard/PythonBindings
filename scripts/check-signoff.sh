#!/bin/bash
# commit-msg hook: the DCO check on pull requests (the probot/dco GitHub App,
# a required status check on main) requires a Signed-off-by trailer that
# exactly matches the commit author. Catch a mismatch here, before it reaches
# that PR check. See the "How to contribute" section of CONTRIBUTING.md.
#
# Install it as a local git hook:
#   cp scripts/check-signoff.sh .git/hooks/commit-msg
#   chmod +x .git/hooks/commit-msg
#
# Adapted from the same check in OAAX-standard/reference-implementation.
set -euo pipefail

msg_file=$1
name=${GIT_AUTHOR_NAME:-$(git config user.name)}
email=${GIT_AUTHOR_EMAIL:-$(git config user.email)}
expected="Signed-off-by: ${name} <${email}>"

if grep -qF "$expected" "$msg_file"; then
  exit 0
fi

echo "DCO: sign-off missing or doesn't match the commit author." >&2
echo "  expected: $expected" >&2
found=$(grep '^Signed-off-by:' "$msg_file" || true)
if [ -n "$found" ]; then
  echo "  found:    $found" >&2
fi
echo "Commit with 'git commit -s' using your own git identity." >&2
exit 1
