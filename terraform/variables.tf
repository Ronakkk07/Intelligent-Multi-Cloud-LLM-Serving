variable "project_name" {
  description = "Prefix for all resource names"
  type        = string
  default     = "llm-router"
}

variable "aws_region" {
  description = "AWS region for EKS cluster"
  type        = string
  default     = "us-east-1"
}

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "azure_location" {
  description = "Azure region for AKS cluster"
  type        = string
  default     = "eastus"
}

variable "hf_token" {
  description = "HuggingFace API token for downloading Mistral-7B and Llama-2-13B"
  type        = string
  sensitive   = true
}

variable "eks_gpu_instance_type" {
  description = "EC2 instance type for GPU node group"
  type        = string
  default     = "g5.xlarge"   # 1x NVIDIA A10G, 24GB VRAM, ~$1.006/hr on-demand
}

variable "aks_gpu_vm_size" {
  description = "Azure VM size for GPU node pool"
  type        = string
  default     = "Standard_NV6ads_A10_v5"  # 1x NVIDIA A10, 6 vCPU, ~$0.454/hr on-demand
}

variable "gpu_node_min" {
  description = "Minimum GPU nodes per cluster (0 = scale to zero when idle)"
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  description = "Maximum GPU nodes per cluster"
  type        = number
  default     = 2
}

variable "gpu_node_desired" {
  description = "Initial GPU node count"
  type        = number
  default     = 1
}
