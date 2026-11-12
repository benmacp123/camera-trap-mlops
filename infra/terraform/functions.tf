locals {
  # No functions until the image is pushed (see var.deploy_functions).
  sizes = var.deploy_functions ? toset([for size in var.memory_sizes : tostring(size)]) : toset([])
}

# The image by digest, so pushing a rebuilt image and applying again
# updates the functions.
data "aws_ecr_image" "champion" {
  count           = var.deploy_functions ? 1 : 0
  repository_name = aws_ecr_repository.lambda.name
  image_tag       = var.image_tag
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name_prefix}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

# Least privilege: read crops, write results, write its own logs.
data "aws_iam_policy_document" "lambda" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.crops.arn}/crops/*"]
  }

  statement {
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.crops.arn}/runs/*"]
  }

  statement {
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:*:*:log-group:/aws/lambda/${var.name_prefix}-*:*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.name_prefix}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# Created here, not by Lambda on first use, so the logs expire and
# `destroy` removes them.
resource "aws_cloudwatch_log_group" "lambda" {
  for_each          = local.sizes
  name              = "/aws/lambda/${var.name_prefix}-champion-${each.key}"
  retention_in_days = 7
}

# One function per memory size, all running the same image. A function
# that is never invoked costs nothing.
resource "aws_lambda_function" "champion" {
  for_each      = local.sizes
  function_name = "${var.name_prefix}-champion-${each.key}"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda.repository_url}@${data.aws_ecr_image.champion[0].image_digest}"
  architectures = ["x86_64"]
  memory_size   = tonumber(each.key)

  # The maximum. A shard of 256 crops takes a few minutes at most; the
  # driver's shard size and concurrency are what bound the spend.
  timeout = 900

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy.lambda]
}
