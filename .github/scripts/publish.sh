#!/usr/bin/env bash
# publish.sh <feed> <source-dir>  -- replace <feed>/ on the orphan branch `data`, keep every other feed, one commit, force push.
set -euo pipefail
feed="$1"; src="$(cd "$2" && pwd)"
repo="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
pub="$(mktemp -d)"
if git clone -q --depth 1 --branch data "$repo" "$pub" 2>/dev/null; then :; else
  git -C "$pub" init -q -b data
  git -C "$pub" remote add origin "$repo"
fi
cd "$pub"
rm -rf "$feed"; mkdir -p "$feed"; cp -r "$src"/. "$feed"/
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git checkout -q --orphan publish
git add -A
git commit -q -m "data $(date -u +%Y-%m-%dT%H:%MZ) ($feed)"
git push -q -f origin publish:data
