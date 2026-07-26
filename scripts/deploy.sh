#!/bin/bash
# Deploy ornith-flight to GitHub

set -e

echo "🚀 Deploying ornith-flight to GitHub"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Install it first:"
    echo "   brew install gh"
    echo "   gh auth login"
    exit 1
fi

# Check authentication
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Run:"
    echo "   gh auth login"
    exit 1
fi

echo "✅ GitHub CLI authenticated"
echo ""

# Navigate to repository
cd /Users/saiduttaabhishekdash/ornith-flight

# Create GitHub repository
echo "📦 Creating GitHub repository..."
gh repo create ornith-flight \
    --public \
    --source=. \
    --description="Expert streaming for Ornith 35B MoE on consumer hardware" \
    --push

echo ""
echo "✅ Repository created and pushed!"
echo ""

# Add topics
echo "🏷️  Adding topics..."
gh repo edit --add-topic machine-learning
gh repo edit --add-topic moe
gh repo edit --add-topic inference
gh repo edit --add-topic quantization
gh repo edit --add-topic python
gh repo edit --add-topic deep-learning

echo ""
echo "✅ Topics added!"
echo ""

# Enable features
echo "⚙️  Enabling repository features..."
gh repo edit --enable-issues
gh repo edit --enable-discussions

echo ""
echo "✅ Features enabled!"
echo ""

# Get repository URL
REPO_URL=$(gh repo view --json url -q .url)

echo "🎉 Deployment complete!"
echo ""
echo "Repository: $REPO_URL"
echo ""
echo "Next steps:"
echo "1. Visit $REPO_URL"
echo "2. Create a release: gh release create v0.1.researchtype --title 'Prototype Complete'"
echo "3. Star your own repo 😄"
