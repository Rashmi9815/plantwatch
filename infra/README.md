# PlantWatch infrastructure (Terraform)

Provisions the serving stack: an S3 artifact bucket, an IAM role, a Python Lambda that
scores sensor windows, and an HTTP API Gateway route (`POST /score`) in front of it.

The same `.tf` runs against **LocalStack** (free, local) and **real AWS** — only the
command changes.

## 1. Build the Lambda package

```bash
make lambda-zip        # -> build/lambda.zip (trained model + source + deps)
```

## 2a. Deploy to LocalStack (recommended for development)

Needs Docker running and the `terraform-local` wrapper (`pip install terraform-local`).

```bash
make up                # start LocalStack (from repo root)
make tf-local          # tflocal init && tflocal apply -auto-approve
```

`tflocal` injects the `http://localhost:4566` endpoints and test credentials, so the
plain `.tf` files stay clean and real-AWS-ready.

## 2b. Deploy to real AWS (for screenshots, then tear down)

```bash
cd infra
terraform init
terraform apply        # review the plan, then approve
# ... take your screenshots ...
terraform destroy      # IMPORTANT: stop the meter
```

## Test the endpoint

```bash
curl -X POST "$(terraform -chdir=infra output -raw score_endpoint)" \
  -H 'content-type: application/json' \
  -d '{"machine_id":"SYNTH-1","window":[[50,60,70,80,90], ...30 rows...]}'
# -> {"machine_id":"SYNTH-1","model_version":"pca_reconstruction@w30","score":...,"is_anomaly":false}
```

## Cost note

Lambda + API Gateway + a small S3 bucket are nearly free at demo volume, but **always
`terraform destroy`** after a real-AWS session and keep an AWS Budgets alert set.
