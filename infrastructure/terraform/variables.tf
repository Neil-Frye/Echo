variable "aws_region" {
  description = "AWS region to deploy to"
  default     = "us-east-1"
  type        = string
}

variable "environment" {
  description = "Environment (development, staging, production)"
  default     = "development"
  type        = string
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}