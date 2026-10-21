# Artifact / data-lake bucket. In production the Lambda would load its model from
# here; in this skeleton the model ships inside the deployment package for simplicity.
resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifact_bucket
}
