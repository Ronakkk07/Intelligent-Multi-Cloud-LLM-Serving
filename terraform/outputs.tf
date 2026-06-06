output "eks_cluster_name" {
  description = "EKS cluster name — use in aws eks update-kubeconfig"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS API server endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_kubeconfig_command" {
  description = "Run this to configure kubectl for EKS"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "aks_cluster_name" {
  description = "AKS cluster name"
  value       = azurerm_kubernetes_cluster.aks.name
}

output "aks_resource_group" {
  description = "Azure resource group containing the AKS cluster"
  value       = azurerm_resource_group.rg.name
}

output "aks_kubeconfig_command" {
  description = "Run this to configure kubectl for AKS"
  value       = "az aks get-credentials --resource-group ${azurerm_resource_group.rg.name} --name ${azurerm_kubernetes_cluster.aks.name}"
}

output "next_steps" {
  description = "What to do after terraform apply"
  value       = <<-EOT
    After 'terraform apply':
    1. Run both kubeconfig commands above to register the clusters.
    2. Switch to EKS context and apply k8s manifests:
         kubectl config use-context <eks-arn>
         kubectl apply -f ../k8s/namespace.yaml
         kubectl create secret generic hf-token --from-literal=token=<HF_TOKEN> -n llm-serving
         kubectl apply -f ../k8s/vllm-mistral-aws.yaml
         kubectl apply -f ../k8s/vllm-llama-aws.yaml
         kubectl apply -f ../k8s/hpa.yaml
    3. Switch to AKS context and repeat for Azure:
         kubectl config use-context <aks-name>
         kubectl apply -f ../k8s/namespace.yaml
         kubectl create secret generic hf-token --from-literal=token=<HF_TOKEN> -n llm-serving
         kubectl apply -f ../k8s/vllm-mistral-azure.yaml
         kubectl apply -f ../k8s/vllm-llama-azure.yaml
         kubectl apply -f ../k8s/hpa.yaml
    4. Get LoadBalancer IPs and update config/endpoints.yaml.
    5. Run: python -m experiments.run_real_experiment --live
  EOT
}
