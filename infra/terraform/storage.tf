# Crops in, logits out. Private: the crops are public LILA data, but
# nothing here needs to be readable by anyone but the function and the owner.
resource "aws_s3_bucket" "crops" {
  bucket_prefix = "${var.name_prefix}-crops-"

  # Lets `destroy` remove the bucket with its objects still in it.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "crops" {
  bucket                  = aws_s3_bucket.crops.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "crops" {
  bucket = aws_s3_bucket.crops.id

  rule {
    id     = "expire-everything"
    status = "Enabled"

    filter {}

    expiration {
      days = var.expire_days
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# Holds the container image. Created by the first apply, before the
# functions, because the image must be pushed before a function can use it.
resource "aws_ecr_repository" "lambda" {
  name = "${var.name_prefix}-lambda"

  # Lets `destroy` remove the repository with the image still in it.
  force_delete = true
}
