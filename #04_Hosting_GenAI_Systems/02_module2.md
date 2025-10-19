# 5. Business & Financial Considerations

## 5.1 Total Cost of Ownership (TCO) Model — Templates & Inputs

| **Component** | **CapEx (One-time)** | **OpEx (Recurring)** | **Notes / Inputs** |
|----------------|----------------------|----------------------|--------------------|
| GPU/Compute Hardware | \$300K–\$1M | – | Depends on GPU count, type (A100/H100), vendor. |
| Storage Infrastructure | \$50K–\$150K | – | NVMe + Object storage cluster. |
| Networking Gear | \$30K–\$100K | – | Switches, routers, firewalls, cabling. |
| Datacenter Space / Rack Rental | – | \$1K–\$4K per rack/month | Includes power, cooling, SLA. |
| Electricity / Cooling | – | \$2K–\$10K per rack/month | Highly variable by region. |
| Software / Licensing | \$10K–\$50K | \$2K–\$5K/year | Monitoring, orchestration, etc. |
| Staff / Engineers | – | \$200K–\$500K/year | Platform SREs, ML engineers, infra admins. |

---

## 5.2 CapEx vs OpEx Scenarios — 3- and 5-Year Models

| **Scenario** | **3-Year Cost** | **5-Year Cost** | **Remarks** |
|---------------|----------------|----------------|--------------|
| Cloud API (Usage-based) | \$1.2M | \$2.4M | At 10M token/day average. Scales linearly. |
| On-Prem Colocated (CapEx-heavy) | \$850K | \$1.1M | Lower long-term TCO beyond year 3. |
| Hybrid (Training on cloud + local inference) | \$950K | \$1.3M | Best balance between control & agility. |

---

## 5.3 Unit Economics: Cost per Inference / Token / Epoch

| **Metric** | **Cloud (API)** | **On-Prem (GPU farm)** | **Notes** |
|-------------|----------------|-------------------------|-----------|
| Cost per 1K tokens (inference) | \$0.002–\$0.01 | \$0.0005–\$0.001 | Depends on utilization and power cost. |
| Cost per fine-tuning epoch | \$100–\$250 | \$20–\$50 | Amortized over hardware life. |
| Cost per inference request (avg.) | \$0.001 | \$0.0002 | Up to 80% cheaper at scale. |

---

## 5.4 Opportunity Cost of Slower Innovation vs Cloud Agility

| **Factor** | **Cloud Advantage** | **On-Prem Trade-off** |
|-------------|--------------------|------------------------|
| Experimentation Speed | Instant provisioning | Weeks to deploy hardware |
| Model Refresh Cycles | Auto-updates via API | Manual deployment/testing |
| R&D Agility | Rapid iteration | Requires infra coordination |
| Vendor Features | Built-in integrations | Need self-hosted equivalents |

---

## 5.5 Procurement, Leasing, and Financing Options

| **Option** | **Pros** | **Cons** |
|-------------|----------|----------|
| Direct Purchase (CapEx) | Full control, tax depreciation | High upfront cost |
| Leasing / Rent-to-Own | Spread payments over 3–5 years | Slightly higher total cost |
| Cloud GPU Marketplace (spot rental) | Flexible scaling | Less control, volatile pricing |
| Vendor Financing Programs | Deferred payments, support bundles | Locked with specific vendor |

---

## 5.6 Pricing Models for Customers

| **Model** | **Description** | **Revenue Implications** |
|------------|-----------------|---------------------------|
| On-Prem Managed Service | Customer owns hardware, provider manages stack | Recurring service revenue |
| SaaS Hybrid | Customer uses remote API + optional on-prem node | Subscription-based revenue |
| Consulting / Integration | One-time setup, tuning, and MLOps enablement | High-margin project-based |
| GPU Sharing Platform | Shared resource billing (token/hour-based) | Recurring with utilization scaling |

---

## 5.7 Cost Optimization Levers

| **Lever** | **Impact** | **Techniques / Tools** |
|------------|------------|------------------------|
| Utilization Scheduling | +30–50% ROI | Queue batching, job schedulers (KubeFlow, Slurm) |
| Model Distillation | -40% compute cost | Use smaller distilled model versions |
| Quantization / Pruning | -30% memory footprint | INT8/FP8 inference |
| Power Optimization | -10–15% OpEx | GPU undervolting, dynamic throttling |
| Multi-Tenancy / Batching | +2× throughput | Model serving frameworks (vLLM, Triton) |

---

# 6. Compliance, Legal & Data Governance

## 6.1 Data Residency and Jurisdiction Mapping

| **Region** | **Primary Concern** | **Recommendation** |
|-------------|--------------------|---------------------|
| EU | GDPR, data export control | Keep PII data in-region |
| India | DPDP Act, RBI guidelines | Use India-hosted DCs |
| US | HIPAA, PCI-DSS | Segmented clusters for regulated workloads |
| Middle East | Sovereignty clauses | Government-approved hosting zones |

---

## 6.2 Regulatory Frameworks Checklist

- GDPR — Data minimization, purpose limitation, RTBF compliance  
- HIPAA — PHI encryption, access audit, breach reporting  
- PCI-DSS — Encrypted payment data storage  
- SOC 2 Type II — Logging, monitoring, control validation  
- ISO 27001 — Security governance baseline  

---

## 6.3 Contracts & SLAs for Rented Datacenter Space

| **Clause** | **Purpose** | **Typical Requirement** |
|-------------|-------------|--------------------------|
| Power & Cooling SLA | Guarantee uptime | ≥ 99.9% availability |
| Physical Access | Security control | Badge + biometric + CCTV |
| Incident Response | Fault isolation | 4-hour onsite support |
| Data Privacy Clauses | Compliance | NDAs, jurisdictional coverage |

---

## 6.4 Audit Trails, Forensics, and E-Discovery

- Maintain immutable logs (e.g., WORM storage).  
- Audit model access & data lineage.  
- Chain-of-custody for all datasets & model checkpoints.  
- Automated SIEM integration (Splunk, ELK, Chronicle).  

---

## 6.5 Data Lifecycle Policies

| **Stage** | **Policy** | **Tools / Notes** |
|------------|------------|------------------|
| Ingest | Validate + anonymize | DLP, schema enforcement |
| Storage | Encrypt at rest | AES-256, KMS-managed |
| Access | RBAC & purpose limitation | IAM, audit logs |
| Retention | 1–5 years configurable | Object lock |
| Deletion | Verified purge | Cryptographic erase |

---

## 6.6 Legal Considerations — Model IP & Fine-Tuning

- Ensure IP assignment from data owners/customers.  
- Define ownership of fine-tuned weights in contract.  
- Handle derivative works under AI model licensing (e.g., LLaMA 2 Community License).  

---

## 6.7 Privacy Enhancing Technologies (PETs)

| **Technique** | **Use Case** | **Impact** |
|----------------|--------------|------------|
| Differential Privacy | Protecting sensitive training data | Limits memorization of private info |
| Homomorphic Encryption | Federated training | Secure computation without decryption |
| Trusted Execution Environments (TEE) | Inference isolation | Intel SGX, AMD SEV, Azure Confidential VMs |
| Federated Learning | Multi-tenant collaboration | Decentralized updates with DP |
| Synthetic Data Generation | Privacy-preserving training | Use GANs / VAEs for mock datasets |

---

# 7. Security Architecture (Detailed)

## 7.1 Network Segmentation & Zero Trust

- Micro-segmentation by tenant or project.  
- Enforce mTLS and identity-based network policies.  
- Isolate inference from training traffic.  
- Use service mesh (Istio, Linkerd) for authentication.  

---

## 7.2 Threat Models: Perimeter vs Internal

| **Type** | **Attack Vector** | **Mitigation** |
|-----------|------------------|----------------|
| Perimeter | Phishing, credential theft, exposed ports | MFA, VPN, bastion access |
| Internal | Rogue employee, insider privilege abuse | RBAC, just-in-time credentials |
| Model-specific | Prompt injection, data exfiltration | Input sanitization, guardrails |

---

## 7.3 Secrets Management and Key Lifecycle

| **Best Practice** | **Implementation** |
|--------------------|-------------------|
| Store secrets centrally | HashiCorp Vault, AWS KMS |
| Rotate keys regularly | 30/90-day rotation |
| Encrypt in transit & rest | TLS 1.3, AES-256 |
| Audit key access | Centralized logging |

---

## 7.4 Supply Chain Security

- Signed model artifacts (cosign, sigstore).  
- Hash verification before deployment.  
- Immutable container registry with provenance metadata.  

---

## 7.5 Runtime Protections

| **Layer** | **Control** | **Tools / Examples** |
|------------|-------------|----------------------|
| Host | SELinux, AppArmor | OS-level confinement |
| Container | Read-only FS, seccomp, non-root | Docker, PodSecurityPolicy |
| Network | Mutual TLS | Istio, SPIFFE/SPIRE |
| Role Management | RBAC, ABAC | Kubernetes RBAC |

---

## 7.6 Adversarial & Poisoning Threats

| **Threat** | **Vector** | **Mitigation** |
|-------------|------------|----------------|
| Data Poisoning | Tampered datasets | Hash checks, data validation |
| Model Extraction | Query scraping | Rate limiting, watermarking |
| Prompt Injection | Malicious inputs | Context filters, instruction guarding |
| Backdoor Inference | Malicious fine-tune | Secure model provenance |

---

## 7.7 Incident Response Playbook

1. Detect anomaly via SIEM alerts.  
2. Isolate affected model endpoint.  
3. Validate artifact integrity via signatures.  
4. Rotate access tokens & keys.  
5. Run post-mortem RCA with model audit logs.  

---

## 7.8 Secure Model Deployment

- Only signed model binaries and containers.  
- Maintain provenance metadata (who trained, dataset version).  
- Use Kubernetes admission controller to enforce integrity.  

---

# 8. Infrastructure: Hardware & Datacenter Basics

## 8.1 GPU / Accelerator Choices

| **Vendor** | **Model** | **Strength** | **Use Case** |
|-------------|------------|--------------|--------------|
| NVIDIA | A100 / H100 | Best overall performance | Training & inference |
| AMD | MI250 / MI300 | Cost-effective | Mixed workloads |
| Intel Habana | Gaudi 2 | Efficient training | Custom ASIC |
| Cerebras | CS-2 | Large-scale model training | Extreme-scale LLMs |
| Graphcore | IPU Pod | Graph-centric AI | Sparse workloads |

---

## 8.2 CPU vs GPU Tradeoffs

| **Task** | **Best Processor** | **Rationale** |
|-----------|-------------------|---------------|
| Training | GPU / TPU | Massive parallelism |
| Fine-tuning | GPU | Moderate parallelism |
| Inference (light) | CPU | Low cost per req |
| Inference (batch) | GPU | Better throughput |

---

## 8.3 Memory, NVMe & IO

- Use NVLink/PCIe Gen 5 for GPU-GPU communication.  
- At least 1.5× model size in VRAM for inference.  
- NVMe RAID for fast checkpointing.  

---

## 8.4 Network Fabrics: Ethernet vs InfiniBand

| **Fabric** | **Bandwidth** | **Latency** | **Use Case** |
|-------------|---------------|--------------|---------------|
| Ethernet (100G) | 100 Gbps | ~10 µs | General workloads |
| InfiniBand HDR/NDR | 200–400 Gbps | < 2 µs | Distributed training |
| RDMA over Converged Ethernet | 100–200 Gbps | < 5 µs | Hybrid DC workloads |

---

## 8.5 Rack Power & Cooling

| **Factor** | **Typical Range** | **Notes** |
|-------------|------------------|------------|
| Power draw per rack | 10–50 kW | Depends on GPU density |
| Cooling | Air / Liquid | Liquid for > 30 kW racks |
| UPS / Generator | N+1 redundancy | Minimum Tier III DC |

---

## 8.6 Physical Security & Colocation Contracts

- Access control: RFID, biometric, 24×7 CCTV.  
- Visitor logging, escorted access.  
- SLA: 99.9% uptime, 4-hour hardware replacement.  

---

## 8.7 Hardware Lifecycle & Spares

- GPU refresh: every 3 years (or when <70% perf/watt).  
- Maintain 10–15% spare nodes for failover.  
- Track via asset inventory + predictive RMA.  

---

## 8.8 Hot vs Cold Aisle Containment

| **Cooling Type** | **Density (kW/rack)** | **Best For** |
|-------------------|----------------------|---------------|
| Hot Aisle | < 20 kW | Standard air-cool |
| Cold Aisle | 20–30 kW | Moderate GPU load |
| Liquid Immersion | > 30 kW | Dense GPU clusters |

---

# 9. Storage & Data Architecture

## 9.1 Storage Tiering

| **Tier** | **Media** | **Use Case** |
|-----------|-----------|---------------|
| Hot | NVMe | Active training/inference |
| Warm | SSD | Preprocessed data |
| Cold | HDD / Object Storage | Archival, logs |

---

## 9.2 Latency-Sensitive vs Batch Datasets

- Split pipelines for inference vs training.  
- Use local cache layer (Redis, NVMe) for active data.  
- Asynchronous data loaders for batch ETL.  

---

## 9.3 Dataset Versioning & Catalog

| **Tool** | **Purpose** |
|-----------|-------------|
| DVC | Dataset version control |
| LakeFS | Git-like data branching |
| Pachyderm | Data pipeline automation |

---

## 9.4 Backup, Replication, & Disaster Recovery

- 3-2-1 backup policy (3 copies, 2 media, 1 offsite).  
- Asynchronous replication between DCs.  
- Test DR recovery quarterly.  

---

## 9.5 Secure Ingestion Pipelines

- TLS in transit, AES-256 at rest.  
- Checksum verification for data integrity.  
- Data validation at ingest via schema registry.  

---

## 9.6 Model Artifact Storage

- Central model registry (MLflow, Weights & Biases, or self-hosted).  
- Signed blobs & checksum verification.  
- Store checkpoints and configs with metadata manifest.  

---

# 10. Networking & Connectivity

## 10.1 Cross-Datacenter Links

| **Metric** | **Recommendation** |
|-------------|--------------------|
| Bandwidth | ≥ 40 Gbps per link |
| Redundancy | Dual-path fiber or MPLS |
| Latency | < 10 ms between sites |

---

## 10.2 Peering & Hybrid Connectivity

- Private peering with cloud (AWS DX, Azure ExpressRoute).  
- Use VPN tunnels or SD-WAN for hybrid orchestration.  
- Ensure QoS and packet marking for inference traffic.  

---

## 10.3 Bandwidth Budgeting

| **Task** | **Bandwidth Need** | **Comment** |
|-----------|--------------------|--------------|
| Training dataset upload | 5–20 Gbps | High I/O |
| Model checkpoint sync | 1–5 Gbps | Frequent writes |
| Inference API calls | < 1 Gbps | Latency-sensitive |

---

## 10.4 Load Balancing & Gateways

- Use L7 API Gateway (Kong, NGINX, Envoy).  
- Layer 4 LB for gRPC inference services.  
- Sticky sessions for long-context models.  

---

## 10.5 Monitoring Network Performance

- NetFlow / sFlow telemetry.  
- Prometheus + Grafana dashboards.  
- Alert on packet loss, RTT > 100 ms.  
- QoS tagging for priority inference pipelines.  

---
