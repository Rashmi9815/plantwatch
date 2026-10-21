# Observability for the serving stack: a log group, two alarms, and a dashboard.

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${aws_lambda_function.scorer.function_name}"
  retention_in_days = 14
}

# Operational health: the function is throwing errors.
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project}-lambda-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  dimensions = {
    FunctionName = aws_lambda_function.scorer.function_name
  }
}

# Model health: anomaly rate (from the EMF AnomalyFlag metric) is sustained high,
# which usually means drift or a genuine plant problem worth paging on.
resource "aws_cloudwatch_metric_alarm" "high_anomaly_rate" {
  alarm_name          = "${var.project}-high-anomaly-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "AnomalyFlag"
  namespace           = "PlantWatch"
  period              = 300
  statistic           = "Average"
  threshold           = 0.3
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project}-serving"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6,
        properties = {
          title   = "Anomaly rate",
          region  = var.region,
          stat    = "Average",
          period  = 300,
          metrics = [["PlantWatch", "AnomalyFlag"]]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6,
        properties = {
          title   = "Mean anomaly score",
          region  = var.region,
          stat    = "Average",
          period  = 300,
          metrics = [["PlantWatch", "AnomalyScore"]]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 24, height = 6,
        properties = {
          title  = "Lambda invocations & errors",
          region = var.region,
          stat   = "Sum",
          period = 60,
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.scorer.function_name],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.scorer.function_name]
          ]
        }
      }
    ]
  })
}
