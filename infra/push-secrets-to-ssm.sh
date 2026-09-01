#!/usr/bin/env bash
# Pushes the LLM provider API keys from the local .env into AWS SSM Parameter
# Store, as SecureString, so CI (via its OIDC role) can fetch them at runtime
# instead of using GitHub Actions Secrets. Langfuse's keys are NOT pushed here --
# infra/langfuse-ec2.yaml's own boot-time UserData already publishes those
# (/langfuse-ec2/base-url, /langfuse-ec2/public-key, /langfuse-ec2/secret-key)
# the moment that EC2 stack comes up, since they're generated fresh on that
# instance, not read from this repo's .env.
#
# Never echoes the actual secret values to stdout.
#
# Usage:
#   ./infra/push-secrets-to-ssm.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env file found at $ENV_FILE -- nothing to push." >&2
  exit 1
fi

# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

REGION="${AWS_REGION:-$(aws configure get region)}"

push() {
  local name="$1" value="$2"
  if [[ -z "$value" ]]; then
    echo "  SKIP $name -- empty in .env"
    return
  fi
  aws ssm put-parameter \
    --region "$REGION" \
    --name "$name" \
    --type SecureString \
    --overwrite \
    --value "$value" >/dev/null
  echo "  OK   $name"
}

echo "Pushing LLM provider keys to SSM Parameter Store (region: $REGION)..."
push "/dubba/anthropic-api-key" "${ANTHROPIC_API_KEY:-}"
push "/dubba/groq-api-key" "${GROQ_API_KEY:-}"

echo ""
echo "Done. Langfuse keys are published separately by infra/langfuse-ec2.yaml's"
echo "own boot script (/langfuse-ec2/base-url, /langfuse-ec2/public-key,"
echo "/langfuse-ec2/secret-key) -- not touched by this script."
echo ""
echo "Full parameter set eval-gate will read at CI runtime:"
echo "  /dubba/anthropic-api-key"
echo "  /dubba/groq-api-key"
echo "  /langfuse-ec2/base-url"
echo "  /langfuse-ec2/public-key"
echo "  /langfuse-ec2/secret-key"
