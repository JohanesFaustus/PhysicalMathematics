#!/bin/bash
TAG="v$(date +%Y.%m.%d)"
PDF="Main.pdf"

gh release create "$TAG" "$PDF" \
  --title "Physical Mathematics $TAG" \
  --notes "Auto-generated release on $(date)"
