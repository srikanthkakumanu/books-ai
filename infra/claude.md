# Infrastructure — Claude Code Context

## Purpose

Docker Compose for local development, Kubernetes manifests for staging/prod, and Terraform for AWS cloud resources. All infra is code — no manual console changes.

## Directory Layout

```
infra/
├── claude.md
├── docker/
│   ├── docker-compose.dev.yml      # Local dev: all services + dependencies
│   └── docker-compose.test.yml     # CI test environment
├── k8s/
│   ├── base/                       # Kustomize base manifests
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── book-service/
│   │   ├── user-service/
│   │   ├── agents/
│   │   ├── mcp-server/
│   │   └── ...
│   └── overlays/
│       ├── dev/                    # Dev cluster overrides
│       └── prod/                   # Prod cluster overrides
└── terraform/
    ├── modules/
    │   ├── rds/                    # PostgreSQL RDS instance
    │   └── eks/                    # EKS cluster
    └── envs/
        ├── dev/
        └── prod/
```

## Local Dev Stack

`docker-compose.dev.yml` starts:

| Service | Port | Purpose |
|---------|------|---------|
| postgres | 5432 | PostgreSQL 16 + pgvector |
| redis | 6379 | Cache + sessions |
| kafka | 9092 | Event bus |
| zookeeper | 2181 | Kafka dependency |
| jaeger | 16686 | Distributed tracing UI |
| adminer | 8080 | DB admin UI |

Python services and the Next.js UI are run locally (hot-reload) — not in Docker during development.

## K8s Conventions

- Namespace: `books-ai-{env}` (e.g. `books-ai-prod`)
- All workloads use `Deployment` (not bare pods)
- Resource requests and limits on every container — no exceptions
- Secrets from AWS Secrets Manager via External Secrets Operator — never `kubectl create secret`
- Liveness probe: `GET /health` (fast, no DB check)
- Readiness probe: `GET /health/ready` (checks DB connectivity)
- PodDisruptionBudget on every Deployment with `minAvailable: 1`

## Terraform Rules

- State stored in S3 + DynamoDB lock
- Modules are reusable building blocks (rds, eks, elasticache)
- Environments (`dev`, `prod`) compose modules with different variable values
- Never use `terraform apply` without a `plan` review
- Tag everything: `Project=books-ai`, `Environment={env}`, `ManagedBy=terraform`

## Environment Parity

Dev and prod use the same Docker images, different configs. Environment-specific values come from:
- K8s ConfigMaps (non-sensitive)
- AWS Secrets Manager → External Secrets Operator → K8s Secrets (sensitive)

Never commit `.env` files. Always use `.env.example` with placeholder values.
