output "score_endpoint" {
  description = "POST sensor windows here to get an anomaly decision."
  value       = "${aws_apigatewayv2_api.http.api_endpoint}/score"
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}
