"""Pulumi deploy for agent-service — Cloud Run service."""

import pulumi
import pulumi_gcp as gcp

config = pulumi.Config()
project_id = config.require("project_id")
region = config.get("region") or "us-central1"
image_tag = config.get("image_tag") or "latest"

infra = pulumi.StackReference("organization/infra/dev")
vpc_connector_id = infra.get_output("vpc_connector_id")
agent_api_sa_email = infra.get_output("sa_agent_api")

mcp_server_url = "https://mcp-variant-server-fno64g2krq-uc.a.run.app"

image = f"{region}-docker.pkg.dev/{project_id}/genomic-pipeline/agent-service:{image_tag}"

service = gcp.cloudrunv2.Service(
    "agent-service",
    name="agent-service",
    project=project_id,
    location=region,
    ingress="INGRESS_TRAFFIC_INTERNAL_ONLY",
    template=gcp.cloudrunv2.ServiceTemplateArgs(
        service_account=agent_api_sa_email,
        scaling=gcp.cloudrunv2.ServiceTemplateScalingArgs(
            min_instance_count=1,
            max_instance_count=3,
        ),
        vpc_access=gcp.cloudrunv2.ServiceTemplateVpcAccessArgs(
            connector=vpc_connector_id,
            egress="PRIVATE_RANGES_ONLY",
        ),
        timeout="300s",
        containers=[
            gcp.cloudrunv2.ServiceTemplateContainerArgs(
                image=image,
                resources=gcp.cloudrunv2.ServiceTemplateContainerResourcesArgs(
                    limits={"cpu": "1000m", "memory": "512Mi"},
                ),
                envs=[
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                        name="GCP_PROJECT", value=project_id
                    ),
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                        name="GCP_REGION", value=region
                    ),
                    gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(
                        name="MCP_SERVER_URL", value=mcp_server_url
                    ),
                ],
                liveness_probe=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeArgs(
                    http_get=gcp.cloudrunv2.ServiceTemplateContainerLivenessProbeHttpGetArgs(
                        path="/health",
                    ),
                    initial_delay_seconds=15,
                    period_seconds=30,
                    timeout_seconds=5,
                    failure_threshold=3,
                ),
            )
        ],
    ),
)

# Allow unauthenticated invocation — VPN/internal ingress is the security boundary
gcp.cloudrun.IamMember(
    "agent-service-public",
    project=project_id,
    location=region,
    service=service.name,
    role="roles/run.invoker",
    member="allUsers",
)

pulumi.export("agent_service_name", service.name)
pulumi.export("agent_service_uri", service.uri)
