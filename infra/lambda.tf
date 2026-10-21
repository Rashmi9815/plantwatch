resource "aws_lambda_function" "scorer" {
  function_name    = "${var.project}-scorer"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.11"
  handler          = "plantwatch.serving.handler.handler"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  timeout          = 30
  memory_size      = 1024

  environment {
    variables = {
      PLANTWATCH_MODEL_PATH = "model.joblib" # bundled at the package root
    }
  }
}

# numpy / pandas / scikit-learn are large; for real AWS prefer a Lambda layer or a
# container image rather than a fat zip. scripts/build_lambda.sh builds the simple zip.
