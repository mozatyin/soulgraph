"""Filter operations on SoulGraph — produce sub-graphs by domain or time."""
from __future__ import annotations

from datetime import datetime

from soulgraph.graph.models import SoulGraph


def filter_by_domain(graph: SoulGraph, domain: str) -> SoulGraph:
    items = [item for item in graph.items if domain in item.domains]
    item_ids = {item.id for item in items}
    edges = [edge for edge in graph.edges if edge.from_id in item_ids and edge.to_id in item_ids]
    return SoulGraph(owner_id=graph.owner_id, items=items, edges=edges)


def filter_by_time(graph: SoulGraph, start: datetime, end: datetime) -> SoulGraph:
    items = [item for item in graph.items if item.created_at and start <= item.created_at <= end]
    item_ids = {item.id for item in items}
    edges = [edge for edge in graph.edges if edge.from_id in item_ids and edge.to_id in item_ids]
    return SoulGraph(owner_id=graph.owner_id, items=items, edges=edges)
