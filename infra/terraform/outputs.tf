output "region" {
  description = "Region everything is deployed in."
  value       = var.region
}

output "bucket" {
  description = "Bucket holding crops (crops/) and results (runs/)."
  value       = aws_s3_bucket.crops.bucket
}

output "repository_url" {
  description = "ECR repository the container image is pushed to."
  value       = aws_ecr_repository.lambda.repository_url
}

output "image_tag" {
  description = "Tag the functions' image is pushed under."
  value       = var.image_tag
}

output "functions" {
  description = "Function name for each memory size in MB; empty until deploy_functions is true."
  value       = { for size, function in aws_lambda_function.champion : size => function.function_name }
}
