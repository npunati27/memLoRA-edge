from .config import MAX_GPU_LORA, MAX_CPU_LORA, TIER_TRANSITION_DELAY_MS


class LRUMixin:
    """Tracks local GPU/CPU adapter residency using LRU eviction."""

    def _update_node_tier(self, adapter_name: str, node_ip: str,
                          old_tier: str, new_tier: str):
        """O(1) tier transition for any node in the cluster map."""
        if adapter_name not in self._peer_adapter_state:
            self._peer_adapter_state[adapter_name] = {
                "gpu": set(), "cpu": set(), "disk": set(),
            }
        tiers = self._peer_adapter_state[adapter_name]
        tiers[old_tier].discard(node_ip)
        tiers[new_tier].add(node_ip)

    def _track_local_adapter(self, adapter_name: str):
        """
        Mirror vLLM's LRU eviction logic for the local node.
        Updates both the local LRU and the cluster-wide adapter state map.
        Returns list of (adapter, old_tier, new_tier) changes to broadcast.
        """
        changes = []

        if adapter_name in self._local_gpu_lru:
            self._local_gpu_lru.move_to_end(adapter_name)
            return changes

        if adapter_name in self._local_cpu_lru:
            old_tier = "cpu"
            self._local_cpu_lru.pop(adapter_name)
        else:
            old_tier = "disk"

        if len(self._local_gpu_lru) >= MAX_GPU_LORA:
            evicted_name, _ = self._local_gpu_lru.popitem(last=False)
            self._local_cpu_lru[evicted_name] = None
            self._update_node_tier(evicted_name, self.my_ip, "gpu", "cpu")
            changes.append((evicted_name, "gpu", "cpu"))

            if len(self._local_cpu_lru) > MAX_CPU_LORA:
                cpu_evicted, _ = self._local_cpu_lru.popitem(last=False)
                self._update_node_tier(cpu_evicted, self.my_ip, "cpu", "disk")
                changes.append((cpu_evicted, "cpu", "disk"))

        self._local_gpu_lru[adapter_name] = None
        self._update_node_tier(adapter_name, self.my_ip, old_tier, "gpu")
        changes.append((adapter_name, old_tier, "gpu"))

        return changes

    def _transition_delay_ms(self, old_tier: str, new_tier: str) -> int:
        return TIER_TRANSITION_DELAY_MS.get((old_tier, new_tier), 0)

    def _sum_transition_delay(self, changes):
        """
        Sum emulated delay for a list of (adapter, old_tier, new_tier) transitions.
        Returns (total_ms, details) where details are tuples:
        (adapter, old_tier, new_tier, delay_ms).
        """
        total_ms = 0
        details = []
        for adapter, old_tier, new_tier in changes:
            delay_ms = self._transition_delay_ms(old_tier, new_tier)
            if delay_ms > 0:
                total_ms += delay_ms
                details.append((adapter, old_tier, new_tier, delay_ms))
        return total_ms, details

    def _get_local_tier(self, adapter_name: str) -> str:
        """Return current memory tier for an adapter on this node."""
        if adapter_name in self._local_gpu_lru:
            return "gpu"
        if adapter_name in self._local_cpu_lru:
            return "cpu"
        return "disk"
