#!/usr/bin/env bash
# Tears down the entire infra/langfuse-ec2.yaml stack -- EC2 instance, security
# group, IAM role/instance profile, all in one command. Nothing in that stack has
# an external dependency that blocks deletion (unlike an ECR repo with images still
# in it), so a plain stack delete is sufficient -- no manual cleanup steps.
#
# Usage:
#   ./infra/teardown-langfuse-ec2.sh [stack-name]
# stack-name defaults to langfuse-ec2 (matches the deploy command in README.md).

set -euo pipefail

STACK_NAME="${1:-langfuse-ec2}"
REGION="${AWS_REGION:-$(aws configure get region)}"

echo "Tearing down CloudFormation stack '$STACK_NAME' in region '$REGION'..."
aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"

echo "Waiting for deletion to complete (this polls until the stack is gone)..."
aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"

echo "Stack '$STACK_NAME' deleted -- EC2 instance, security group, and IAM role are gone."
echo "Note: /langfuse-ec2/* SSM parameters are NOT part of the stack (written by the"
echo "instance itself at boot, not declared as CFN resources) -- clean those up too:"
echo "  aws ssm delete-parameters --region $REGION --names /langfuse-ec2/base-url /langfuse-ec2/public-key /langfuse-ec2/secret-key /langfuse-ec2/web-login-email /langfuse-ec2/web-login-password"
