# Solution Architecture

## Project
Connected Vehicle Telemetry Platform

## ADR-001: Streaming Technology

Kafka was selected as the primary event-streaming technology.

The engineering team considered both Kafka and a managed cloud-native queue. Kafka was selected because of existing internal expertise and its ability to support high-throughput event processing.

## ADR-002: Vehicle Authentication

The architecture team selected mutual TLS (mTLS) using vehicle-specific certificates.

Each vehicle will receive a unique certificate during manufacturing. The backend will validate the certificate before accepting telemetry data.

This decision satisfies the customer's preference for certificate-based authentication.

## ADR-003: Deployment Architecture

The production platform will run across three availability zones in the Frankfurt cloud region.

Stateless services will be deployed using Kubernetes. Critical stateful services will use managed replication where possible.

## ADR-004: Telemetry Processing

Streaming telemetry will be processed through Kafka-based pipelines.

Batch uploads will use the same downstream processing components after an initial ingestion stage.

The architecture currently targets approximately 40,000 messages per second. Engineering believes this will cover expected initial traffic with additional capacity available through horizontal scaling.