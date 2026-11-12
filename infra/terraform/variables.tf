variable "region" {
  description = "AWS region to deploy into. The price snapshot (data/prices/) is for eu-west-2."
  type        = string
  default     = "eu-west-2"
}

variable "project" {
  description = "Value of the Project tag on every resource."
  type        = string
  default     = "camera-trap-mlops"
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "camera-trap"
}

variable "alert_email" {
  description = "Address for the budget alarm. Set it in terraform.tfvars (git-ignored)."
  type        = string
}

variable "monthly_budget_usd" {
  description = "AWS Budgets alarm threshold for the whole account, in US dollars."
  type        = number
  default     = 5
}

variable "deploy_functions" {
  description = "Whether to create the Lambda functions. False for the first apply, which creates the registry the image is pushed to; true once the image is there."
  type        = bool
  default     = false
}

variable "image_tag" {
  description = "Tag of the pushed container image the functions run."
  type        = string
  default     = "champion"
}

variable "memory_sizes" {
  description = "One function per memory size in MB, for the sweep. Lambda's CPU share grows with memory: one vCPU at 1769 MB, up to six at 10240 MB. New accounts are often capped at 3008 MB."
  type        = list(number)
  default     = [1769, 3008]
}

variable "expire_days" {
  description = "Days after which the bucket deletes its own objects: a backstop in case destroy is forgotten."
  type        = number
  default     = 7
}
