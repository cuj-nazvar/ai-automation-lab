# Customer Requirements

## Project
Connected Vehicle Telemetry Platform

## Customer
European automotive OEM

## Background
The customer is launching a new generation of connected vehicles and requires a cloud-based platform for collecting and processing vehicle telemetry data.

The platform is expected to support approximately 500,000 vehicles at production launch and should be scalable to 2 million vehicles.

## Key Requirements

### Data Ingestion
- Vehicles will upload telemetry data approximately every 30 seconds while online.
- The platform must support both real-time streaming and delayed batch uploads.
- Telemetry data includes vehicle position, battery status, diagnostic information, and selected sensor data.

### Performance
- Real-time telemetry should normally become available to downstream applications within 5 seconds.
- The system should support at least 50,000 telemetry messages per second during peak periods.

### Security
- All communication between vehicles and the cloud must be encrypted.
- Vehicles must authenticate before telemetry data is accepted.
- The customer prefers certificate-based authentication.

### Availability
- Production availability target is 99.95%.
- The platform must operate across multiple availability zones.

### Delivery
The customer expects production readiness by September 15.

A production-readiness review must take place at least two weeks before launch.