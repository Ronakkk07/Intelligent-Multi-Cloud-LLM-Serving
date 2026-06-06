# AWS EKS cluster with GPU-enabled node group (g5.xlarge = 1x A10G)
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  # GPU node group — scale to zero when not running experiments to save cost
  eks_managed_node_groups = {
    gpu = {
      name           = "gpu-nodes"
      instance_types = [var.eks_gpu_instance_type]

      # AL2_x86_64_GPU includes the NVIDIA driver and nvidia-container-runtime
      ami_type = "AL2_x86_64_GPU"

      min_size     = var.gpu_node_min
      max_size     = var.gpu_node_max
      desired_size = var.gpu_node_desired

      disk_size = 100   # GB — model weights need space

      labels = {
        workload-type = "gpu-inference"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # Allow kubectl from anywhere (restrict for production)
  cluster_additional_security_group_ids = []
}

# Store kubeconfig locally for convenience
resource "local_file" "eks_kubeconfig" {
  depends_on = [module.eks]
  content    = <<-EOF
    # Run this to configure kubectl for EKS:
    # aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}
  EOF
  filename   = "${path.module}/../.kubeconfig-eks-hint.txt"
}
