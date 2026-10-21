#!/usr/bin/env bash
# Build the Lambda deployment package: a trained model + the source + runtime deps.
# Heavy scientific deps make this zip large; for real AWS prefer a Lambda layer or a
# container image. Fine for LocalStack and for demonstrating the shape.
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build/lambda"
ZIP="build/lambda.zip"

rm -rf "$BUILD_DIR" "$ZIP"
mkdir -p "$BUILD_DIR"

echo "==> exporting trained model into the package"
python -m plantwatch.pipelines.export --model pca --out "$BUILD_DIR/model.joblib"

echo "==> copying source"
cp -r src/plantwatch "$BUILD_DIR/plantwatch"

echo "==> installing runtime dependencies into the package"
pip install --quiet --target "$BUILD_DIR" numpy pandas scikit-learn joblib

echo "==> zipping"
( cd "$BUILD_DIR" && zip -qr "../lambda.zip" . )

echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
