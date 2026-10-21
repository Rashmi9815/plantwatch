variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "plantwatch"
}

variable "lambda_zip_path" {
  type        = string
  default     = "../build/lambda.zip"
  description = "Deployment package built by scripts/build_lambda.sh (make lambda-zip)."
}

variable "artifact_bucket" {
  type        = string
  default     = "plantwatch-artifacts"
  description = "S3 bucket for model artifacts and raw data."
}
