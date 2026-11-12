# State stays local and git-ignored (*.tfstate): one owner, one account,
# and `terraform destroy` removes everything, so a remote backend would be
# the only resource left behind.
terraform {
  required_version = "~> 1.16"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.69"
    }
  }
}

provider "aws" {
  region = var.region

  # Every resource carries the tag, so the tagging API can show that
  # `destroy` left nothing behind.
  default_tags {
    tags = {
      Project = var.project
    }
  }
}
