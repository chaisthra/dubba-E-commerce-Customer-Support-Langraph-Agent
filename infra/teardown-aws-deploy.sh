#!/usr/bin/env bash
# Tears down BOTH assignment-2.6 stacks -- ALB/ECS first, then network/RDS/ECR.
# Order matters: aws-ecs-alb.yaml's RDSIngressFromTask resource lives in a
# different stack than the RDS security group it modifies, and the ALB/ECS stack
# generally has to go first anyway since aws-network-rds-ecr.yaml's resources
# (subnets, security group) are referenced BY it. Deleting network/RDS/ECR first
# would fail with a dependency error.
#
# ECR's EmptyOnDelete:true (see aws-network-rds-ecr.yaml) means the repo deletes
# cleanly even with images still in it -- no manual `aws ecr batch-delete-image`
# step needed first, unlike a repo without that flag.
#
# Usage:
#   ./infra/teardown-aws-deploy.sh

set -euo pipefail

REGION="${AWS_REGION:-$(aws configure get region)}"

echo "=== Tearing down aws-ecs-alb (ALB, ECS, autoscaling) ==="
aws cloudformation delete-stack --stack-name dubba-ecs-alb --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name dubba-ecs-alb --region "$REGION"
echo "aws-ecs-alb stack deleted."

echo ""
echo "=== Tearing down aws-network-rds-ecr (private subnets, RDS, ECR) ==="
aws cloudformation delete-stack --stack-name dubba-network-rds-ecr --region "$REGION"
aws cloudformation wait stack-delete-complete --stack-name dubba-network-rds-ecr --region "$REGION"
echo "aws-network-rds-ecr stack deleted."

echo ""
echo "Both stacks deleted. Remaining manual cleanup (not CFN-managed, won't be"
echo "removed automatically):"
echo "  - /dubba/database-url SSM parameter (RDS is gone, this value is now dead):"
echo "      aws ssm delete-parameter --region $REGION --name /dubba/database-url"
echo "  - CloudWatch Logs group /ecs/dubba may take a few minutes to show as gone"
echo "    from the console even after the stack delete completes -- this is normal"
echo "    AWS console lag, not a failed teardown."
echo ""
echo "Langfuse EC2 (infra/langfuse-ec2.yaml) and the GitHub OIDC IAM role"
echo "(dubba-github-actions-deploy) are SEPARATE stacks/resources, not touched by"
echo "this script -- see infra/teardown-langfuse-ec2.sh for the former."
