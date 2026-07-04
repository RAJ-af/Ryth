#!/usr/bin/env bash
# Create the v0.2.0 "Model Core" milestone + tracking issues on GitHub.
#
# Requires the GitHub CLI, authenticated once:
#     gh auth login            # (or: export GH_TOKEN=<token with 'repo' scope>)
#
# Then run:
#     scripts/create_v0.2.0_board.sh
#
# Idempotent-ish: re-running re-creates issues, but skips an existing milestone/label.
set -euo pipefail

REPO="${REPO:-RAJ-af/Ryth}"
MILESTONE="v0.2.0"
LABEL="model-core"

command -v gh >/dev/null || { echo "error: gh CLI not found. Install: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: gh not authenticated. Run: gh auth login"; exit 1; }

echo ">> Ensuring milestone '$MILESTONE' exists on $REPO ..."
gh api "repos/$REPO/milestones" \
   -f title="$MILESTONE" \
   -f state="open" \
   -f description="Model core: transformer architecture from scratch (pure PyTorch). ROADMAP Phase 3." \
   >/dev/null 2>&1 || echo "   (milestone already exists — continuing)"

MS_NUM=$(gh api "repos/$REPO/milestones?state=all" \
   --jq ".[] | select(.title==\"$MILESTONE\") | .number")
echo "   milestone #$MS_NUM"

echo ">> Ensuring label '$LABEL' exists ..."
gh label create "$LABEL" --repo "$REPO" --color 6f42c1 \
   --description "v0.2.0 model core" >/dev/null 2>&1 || echo "   (label exists)"

# One issue per task (order matches the v0.2.0 checklist).
titles=(
  "Transformer Config"
  "RoPE (rotary positional embeddings)"
  "RMSNorm"
  "GQA (grouped-query attention)"
  "SwiGLU feed-forward"
  "KV Cache"
  "FlashAttention interface (SDPA / fallback)"
  "Unit Tests"
  "CPU Forward Pass"
  "GPU Forward Pass"
)

echo ">> Creating task issues ..."
for t in "${titles[@]}"; do
  gh issue create --repo "$REPO" \
     --title "$t" \
     --body "Part of the **v0.2.0 Model Core** milestone (ROADMAP Phase 3 — Training Engine builds on this)." \
     --milestone "$MILESTONE" \
     --label "$LABEL" >/dev/null && echo "   + $t"
done

echo ">> Creating tracking issue with checklist ..."
gh issue create --repo "$REPO" \
  --title "v0.2.0 — Model Core (tracking)" \
  --milestone "$MILESTONE" \
  --label "$LABEL" \
  --body "$(cat <<'EOF'
Milestone: **v0.2.0 — Model Core** (from scratch, pure PyTorch)

- [ ] Transformer Config
- [ ] RoPE
- [ ] RMSNorm
- [ ] GQA
- [ ] SwiGLU
- [ ] KV Cache
- [ ] FlashAttention
- [ ] Unit Tests
- [ ] CPU Forward Pass
- [ ] GPU Forward Pass
EOF
)" >/dev/null && echo "   + tracking issue"

echo ">> Done. Milestone: https://github.com/$REPO/milestone/$MS_NUM"
echo "   Issues:    https://github.com/$REPO/issues?q=is%3Aissue+milestone%3A$MILESTONE"
