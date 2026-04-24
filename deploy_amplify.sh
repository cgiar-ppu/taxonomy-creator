#!/bin/bash
# ============================================================================
# PRMS Taxonomy Explorer — AWS Amplify Deployment Script
# Run this with AWS credentials that have Amplify write access
# Usage: AWS_PROFILE=admin ./deploy_amplify.sh
# ============================================================================

set -euo pipefail

REGION="eu-central-1"
APP_NAME="prms-taxonomy"
REPO_URL="https://github.com/cgiar-ppu/taxonomy-creator"
BRANCH="master"
DOMAIN="synapsis-analytics.com"
SUBDOMAIN="prms-taxonomy"

echo "=== Creating Amplify App: ${APP_NAME} ==="

# Step 1: Create the app (manual deploy - no GitHub connection needed initially)
APP_ID=$(aws amplify create-app \
  --name "${APP_NAME}" \
  --description "PRMS Taxonomy Explorer - CGIAR Research Results Knowledge Graph" \
  --platform WEB \
  --region "${REGION}" \
  --custom-rules '[{"source":"/<*>","target":"/index.html","status":"404-200"}]' \
  --enable-branch-auto-build \
  --query 'app.appId' \
  --output text)

echo "Created app: ${APP_ID}"

# Step 2: Create branch
aws amplify create-branch \
  --app-id "${APP_ID}" \
  --branch-name "${BRANCH}" \
  --stage PRODUCTION \
  --enable-auto-build \
  --region "${REGION}" > /dev/null

echo "Created branch: ${BRANCH}"

# Step 3: Deploy the dist/ directory
echo "Deploying dist/ directory..."

# Create a zip of dist/
cd "$(dirname "$0")"
(cd dist && zip -r ../dist.zip .)

# Create deployment
DEPLOY_URL=$(aws amplify create-deployment \
  --app-id "${APP_ID}" \
  --branch-name "${BRANCH}" \
  --region "${REGION}" \
  --query 'zipUploadUrl' \
  --output text)

# Upload the zip
curl -T dist.zip "${DEPLOY_URL}" --silent

# Start the deployment
JOB_ID=$(aws amplify start-deployment \
  --app-id "${APP_ID}" \
  --branch-name "${BRANCH}" \
  --region "${REGION}" \
  --query 'jobSummary.jobId' \
  --output text)

echo "Deployment started: job ${JOB_ID}"
rm -f dist.zip

# Step 4: Set up custom domain
echo "=== Setting up domain: ${SUBDOMAIN}.${DOMAIN} ==="
aws amplify create-domain-association \
  --app-id "${APP_ID}" \
  --domain-name "${DOMAIN}" \
  --sub-domain-settings "[{\"prefix\":\"${SUBDOMAIN}\",\"branchName\":\"${BRANCH}\"}]" \
  --region "${REGION}" > /dev/null 2>&1 || echo "Domain association may already exist, updating..."

# Step 5: Wait for deployment
echo ""
echo "=== Deployment Summary ==="
echo "App ID:      ${APP_ID}"
echo "Default URL: https://${BRANCH}.${APP_ID}.amplifyapp.com"
echo "Custom URL:  https://${SUBDOMAIN}.${DOMAIN} (may take 10-30 min for DNS/SSL)"
echo ""
echo "Check status: aws amplify get-job --app-id ${APP_ID} --branch-name ${BRANCH} --job-id ${JOB_ID} --region ${REGION}"
echo ""
echo "Done! 🎉"
