"""Federated split builders (IID, Dirichlet skew, synthetic domain)."""

from fed_agent.splits.partition import (
    build_dirichlet_split_primary,
    build_domain_hash_split,
    build_iid_split,
    split_payload,
)

__all__ = [
    "build_iid_split",
    "build_dirichlet_split_primary",
    "build_domain_hash_split",
    "split_payload",
]
