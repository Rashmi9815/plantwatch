# Scheduled retraining loop. EventBridge fires on a schedule; the retrain Lambda trains
# a candidate and promotes it only if it beats the live model (promote-on-improvement).
#
# NOTE: real retraining of heavy models belongs in a SageMaker training job / Autopilot,
# which this Lambda would *start* rather than do inline. The Lambda here runs the
# lightweight path and is enough to demonstrate the closed loop on LocalStack.

resource "aws_cloudwatch_event_rule" "retrain_schedule" {
  name                = "${var.project}-retrain-schedule"
  schedule_expression = "rate(1 day)"
}

resource "aws_lambda_function" "retrain" {
  function_name    = "${var.project}-retrain"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.11"
  handler          = "plantwatch.pipelines.retrain.lambda_handler"
  filename         = var.lambda_zip_path
  source_code_hash = filebase64sha256(var.lambda_zip_path)
  timeout          = 300
  memory_size      = 2048

  environment {
    variables = {
      PLANTWATCH_MODEL_PATH   = "model.joblib"
      PLANTWATCH_RETRAIN_MODEL = "pca"
    }
  }
}

resource "aws_cloudwatch_event_target" "retrain" {
  rule = aws_cloudwatch_event_rule.retrain_schedule.name
  arn  = aws_lambda_function.retrain.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.retrain.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.retrain_schedule.arn
}
