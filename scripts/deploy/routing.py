import random

from .config import TIER_RANK


class RoutingMixin:
    """Routing decision logic: baseline (power-of-two-choices) and memory-aware."""

    def _get_node_tier(self, node_ip: str, adapter_name: str | None) -> str:
        if adapter_name is None:
            return "gpu"
        if node_ip == self.my_ip:
            return self._get_local_tier(adapter_name)
        peer_tiers = self._peer_adapter_state.get(adapter_name, {})
        if node_ip in peer_tiers.get("gpu", set()):
            return "gpu"
        if node_ip in peer_tiers.get("cpu", set()):
            return "cpu"
        return "disk"

    def _get_known_queue_lengths(self) -> dict[str, int]:
        queues = {self.my_ip: self._ongoing}
        for ip in self.peer_ips:
            if ip != self.my_ip:
                queues[ip] = self._peer_queue_lengths.get(ip, 0)
        return queues

    def _choose_target_node_baseline(self, adapter_name: str, source_ip: str) -> str:
        all_nodes = [self.my_ip] + [ip for ip in self.peer_ips if ip != self.my_ip]
        if len(all_nodes) == 1:
            return self.my_ip

        sampled = random.sample(all_nodes, 2)
        queues = self._get_known_queue_lengths()
        n1, n2 = sampled
        q1 = queues.get(n1, 0)
        q2 = queues.get(n2, 0)

        if q1 < q2:
            chosen = n1
        elif q2 < q1:
            chosen = n2
        else:
            chosen = random.choice([n1, n2])

        self.metrics.log(
            "baseline_p2c_choice",
            adapter_name=adapter_name,
            source_ip=source_ip,
            sampled_nodes=sampled,
            sampled_queue_lengths={n1: q1, n2: q2},
            chosen_node=chosen,
        )
        return chosen

    def _choose_target_node_memory(self, adapter_name: str, source_ip: str) -> str:
        all_nodes = [self.my_ip] + [ip for ip in self.peer_ips if ip != self.my_ip]
        if len(all_nodes) == 1:
            return self.my_ip

        queues = self._get_known_queue_lengths()
        candidates = []
        for node in all_nodes:
            tier = self._get_node_tier(node, adapter_name)
            qlen = queues.get(node, 0)
            candidates.append((node, tier, qlen))

        best_rank = min(TIER_RANK[tier] for _, tier, _ in candidates)
        best = [c for c in candidates if TIER_RANK[c[1]] == best_rank]

        if len(best) > 1:
            best_q = min(qlen for _, _, qlen in best)
            best = [c for c in best if c[2] == best_q]

        chosen = random.choice(best)[0]

        self.metrics.log(
            "memory_choice",
            adapter_name=adapter_name,
            source_ip=source_ip,
            candidate_nodes=all_nodes,
            candidate_state={
                node: {"tier": tier, "queue_len": qlen}
                for node, tier, qlen in candidates
            },
            chosen_node=chosen,
        )
        return chosen
