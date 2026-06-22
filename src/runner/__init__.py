"""
Runner module for experiment orchestration.

This package defines the execution layer of the framework, responsible for:

- Defining experiments and execution stages
- Building run IDs and experiment metadata
- Constructing runtime contexts (RunContext)
- Executing experiments via a unified runner interface

The runner is intentionally model-, dataset-, and attack-agnostic.
All domain-specific logic is injected via configuration and builders.
"""