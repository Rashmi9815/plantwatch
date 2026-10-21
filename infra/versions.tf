terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

# Run with `terraform` against real AWS, or with `tflocal` against LocalStack
# (tflocal auto-injects the http://localhost:4566 endpoints and test credentials).
provider "aws" {
  region = var.region
}
