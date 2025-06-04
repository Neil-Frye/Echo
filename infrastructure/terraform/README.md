# EthernalEcho Infrastructure

This directory contains Terraform configurations for provisioning the EthernalEcho infrastructure on AWS.

## Resources Created

- VPC with public and private subnets
- EKS cluster for Kubernetes workloads
- RDS PostgreSQL database
- ElastiCache Redis cluster
- S3 bucket for content storage
- CloudFront distribution for content delivery

## Usage

### Prerequisites

- Terraform 1.0 or higher
- AWS CLI configured with appropriate credentials
- S3 bucket for Terraform state (specified in backend configuration)

### Initialization

```bash
terraform init
```

### Planning

```bash
terraform plan -var-file=terraform.tfvars
```

### Application

```bash
terraform apply -var-file=terraform.tfvars
```

### Destruction

```bash
terraform destroy -var-file=terraform.tfvars
```

## Variables

Create a `terraform.tfvars` file with the following variables:

```hcl
aws_region  = "us-east-1"
environment = "development"
db_username = "postgres"
db_password = "securepassword"
```

## Outputs

After successful application, Terraform will output:

- EKS cluster endpoint and security group
- RDS PostgreSQL hostname, port, and username
- Redis endpoint and port
- S3 bucket name for content
- CloudFront distribution ID and domain name

## Environments

The infrastructure can be deployed to different environments by changing the `environment` variable:

- `development`: For development work
- `staging`: For testing and QA
- `production`: For production deployment

Each environment will have isolated resources with appropriate naming.