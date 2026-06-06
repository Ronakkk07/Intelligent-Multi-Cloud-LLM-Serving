# Azure AKS cluster with GPU node pool (NV6ads_A10_v5 = 1x NVIDIA A10)
resource "azurerm_resource_group" "rg" {
  name     = "${var.project_name}-rg"
  location = var.azure_location

  tags = {
    Project     = var.project_name
    Environment = "research"
    ManagedBy   = "terraform"
  }
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.project_name}-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "${var.project_name}-aks"
  kubernetes_version  = "1.29"

  # System node pool (lightweight — no GPU needed)
  default_node_pool {
    name       = "system"
    node_count = 1
    vm_size    = "Standard_D2s_v3"
    os_disk_size_gb = 50
  }

  identity {
    type = "SystemAssigned"
  }

  # Enable OIDC for workload identity (future: access Azure Blob without secrets)
  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  tags = {
    Project     = var.project_name
    Environment = "research"
  }
}

# Separate GPU node pool — scale to zero when not running experiments
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.aks_gpu_vm_size
  os_disk_size_gb       = 100

  enable_auto_scaling = true
  min_count           = var.gpu_node_min
  max_count           = var.gpu_node_max
  node_count          = var.gpu_node_desired

  node_labels = {
    "workload-type" = "gpu-inference"
  }

  node_taints = ["nvidia.com/gpu=true:NoSchedule"]

  tags = {
    Project = var.project_name
  }
}
