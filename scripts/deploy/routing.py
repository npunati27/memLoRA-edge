from __future__ import annotations

import random

from .config import TIER_RANK, RTT_MAX_MS, MAX_QUEUE_LEN, MEMORY_COST, logger


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

    def _compute_cost(self, node_ip: str, adapter_name: str) -> float:
        """
        cost = 0.4 * queue_cost + 0.4 * memory_cost + 0.2 * network_cost

        queue_cost:   normalized queue length [0, 1]
        memory_cost:  tier load penalty {0.0, 0.015, 1.0, inf}
        network_cost: normalized RTT [0, 1], 0 for local node
        """
        if node_ip == self.my_ip:
            queue_len = self._ongoing
        else:
            queue_len = self._peer_queue_lengths.get(node_ip, 0)
        queue_cost = min(queue_len / MAX_QUEUE_LEN, 1.0)

        tier = self._get_node_tier(node_ip, adapter_name)
        memory_cost = MEMORY_COST.get(tier, 1.0)
        if memory_cost == float("inf"):
            return float("inf")

        if node_ip == self.my_ip:
            network_cost = 0.0
        else:
            rtt = self._measured_rtt.get(node_ip, RTT_MAX_MS)
            if rtt == float("inf"):
                return float("inf") 
            network_cost = min(rtt / RTT_MAX_MS, 1.0)

        return 0.4 * queue_cost + 0.4 * memory_cost + 0.2 * network_cost

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
    
    def _choose_target_node_cost(self, adapter_name: str, source_ip: str) -> str:
        all_nodes = [self.my_ip] + [ip for ip in self.peer_ips if ip != self.my_ip]
        if len(all_nodes) == 1:
            return self.my_ip

        costs = {node: self._compute_cost(node, adapter_name) for node in all_nodes}
        reachable = {node: c for node, c in costs.items() if c != float("inf")}

        if reachable:
            # normal case — at least one node has it on gpu/cpu/disk
            best_cost = min(reachable.values())
            best_nodes = [n for n, c in reachable.items() if c == best_cost]
            chosen = random.choice(best_nodes)
            self.metrics.log(
                "cost_choice",
                adapter_name=adapter_name,
                source_ip=source_ip,
                costs=costs,
                chosen_node=chosen,
                chosen_cost=best_cost,
                s3_fallback=False,
            )
            return chosen

        # S3 fallback — nobody has it locally, must fetch from S3
        # pick the node with the lowest queue + best network to minimize
        # total time including S3 download
        logger.warning(
            f"[routing] adapter={adapter_name} not found locally on any node "
            f"— falling back to S3 fetch"
        )

        def s3_cost(node_ip):
            if node_ip == self.my_ip:
                queue_cost = min(self._ongoing / MAX_QUEUE_LEN, 1.0)
                network_cost = 0.0
            else:
                queue_cost = min(self._peer_queue_lengths.get(node_ip, 0) / MAX_QUEUE_LEN, 1.0)
                rtt = self._measured_rtt.get(node_ip, RTT_MAX_MS)
                if rtt == float("inf"):
                    return float("inf")
                network_cost = min(rtt / RTT_MAX_MS, 1.0)
            # memory cost is the same for everyone (S3) so it cancels out
            # just optimize for queue + network
            return 0.67 * queue_cost + 0.33 * network_cost

        s3_costs = {node: s3_cost(node) for node in all_nodes}
        s3_reachable = {n: c for n, c in s3_costs.items() if c != float("inf")}

        if not s3_reachable:
            # every link is dead, nothing we can do
            logger.error(f"[routing] adapter={adapter_name} no reachable nodes at all")
            return self.my_ip

        chosen = min(s3_reachable, key=s3_reachable.get)
        self.metrics.log(
            "cost_choice",
            adapter_name=adapter_name,
            source_ip=source_ip,
            costs=costs,
            chosen_node=chosen,
            chosen_cost=s3_costs[chosen],
            s3_fallback=True,
        )
        return chosen
