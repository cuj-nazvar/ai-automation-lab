# Project Risks

## RISK-001: Peak Throughput

**Status:** Open  
**Severity:** High

Current performance testing has demonstrated stable processing at approximately 38,000 messages per second.

The customer requirement specifies support for 50,000 messages per second during peak periods.

The performance team believes further Kafka partitioning and consumer scaling can close the gap, but this has not yet been demonstrated in testing.

## RISK-002: Certificate Provisioning

**Status:** Open  
**Severity:** Medium

The vehicle manufacturing team has not completed integration with the certificate provisioning service.

If this interface is delayed, end-to-end testing of vehicle authentication may be affected.

## RISK-003: Production Monitoring

**Status:** Mitigating  
**Severity:** Medium

Core infrastructure metrics are available, but several application-level alerts have not yet been defined.

The operations team requires alerting for ingestion latency, rejected vehicle connections, Kafka consumer lag, and failed batch uploads before production launch.

## RISK-004: Load Test Environment

**Status:** Open  
**Severity:** High

The current load-test environment cannot reliably generate more than 40,000 telemetry messages per second.

Infrastructure capacity for a larger test environment has been requested but has not yet been approved.