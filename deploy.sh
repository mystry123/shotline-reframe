#!/usr/bin/env bash
# Deploy a commit of shotline-reframe on this VM, rolling back if it isn't healthy.
#
#   deploy.sh <commit-sha>
#
# Runs as the `reframe` user (CI calls it through /usr/local/bin/shotline-deploy).
# Steps: fetch from GitHub, check out the commit (it must be on origin/main),
# install dependencies, build, restart the service, health check. If the
# health check fails, the previous commit is redeployed the same way.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/shotline-reframe/app}"
SERVICE="${SERVICE:-shotline-reframe}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8090/health}"
HEALTH_TRIES="${HEALTH_TRIES:-45}"            # × 2 s
LOG="${DEPLOY_LOG:-/var/lib/shotline-reframe/deploys.log}"

target="${1:-}"
if [[ ! "$target" =~ ^[0-9a-f]{7,40}$ ]]; then
  echo "usage: deploy.sh <commit-sha>" >&2
  exit 2
fi

# One deploy at a time.
exec 9>"/var/lib/shotline-reframe/deploy.lock"
flock -n 9 || { echo "Another deploy is running" >&2; exit 3; }

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

cd "$APP_DIR"
git fetch --quiet --prune origin
target="$(git rev-parse --verify "${target}^{commit}")"
if ! git merge-base --is-ancestor "$target" origin/main; then
  log "refused $target: not on origin/main"
  exit 4
fi
previous="$(git rev-parse HEAD)"

install_commit() {
  git checkout --quiet --force --detach "$1"
  [[ -x .venv/bin/python ]] || python3.12 -m venv .venv
  .venv/bin/pip install --quiet --disable-pip-version-check -r requirements.txt
  # "Build": byte-compile, which also catches syntax errors before the restart.
  .venv/bin/python -m compileall -q reframe
  sudo /usr/bin/systemctl restart "$SERVICE"
}

healthy() {
  for _ in $(seq "$HEALTH_TRIES"); do
    if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

log "deploying $target (was $previous)"
if install_commit "$target" && healthy; then
  log "deployed $target: $(curl -fsS --max-time 3 "$HEALTH_URL")"
  exit 0
fi

log "health check failed for $target; rolling back to $previous"
sudo /usr/bin/journalctl -u "$SERVICE" -n 30 --no-pager || true
if install_commit "$previous" && healthy; then
  log "rolled back to $previous"
else
  log "ROLLBACK FAILED: $previous is not healthy either"
fi
exit 1
