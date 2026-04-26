import os
from __future__ import annotations

from .config import ADAPTER_PATH, MAX_GPU_LORA, MAX_CPU_LORA, USE_S3_ADAPTERS


class LRUMixin:
    """Tracks local GPU/CPU adapter residency using LRU eviction."""

    def _empty_tier_map(self) -> dict[str, set]:
        return {"gpu": set(), "cpu": set(), "disk": set(), "s3": set()}

    def _ensure_peer_tiers(self, adapter_name: str) -> dict[str, set]:
        if adapter_name not in self._peer_adapter_state:
            self._peer_adapter_state[adapter_name] = self._empty_tier_map()
        else:
            self._peer_adapter_state[adapter_name].setdefault("gpu", set())
            self._peer_adapter_state[adapter_name].setdefault("cpu", set())
            self._peer_adapter_state[adapter_name].setdefault("disk", set())
            self._peer_adapter_state[adapter_name].setdefault("s3", set())
        return self._peer_adapter_state[adapter_name]

    def _get_local_adapter_path(self, adapter_name: str) -> str:
        return os.path.join(ADAPTER_PATH, adapter_name)

    def _get_initial_local_tier(self, adapter_name: str) -> str:
        if os.path.isdir(self._get_local_adapter_path(adapter_name)):
            return "disk"
        if USE_S3_ADAPTERS:
            return "s3"
        return "disk"

    def _get_local_tier_lists(self, adapter_names: list[str] | None = None) -> dict[str, list[str]]:
        names = adapter_names or self.lora_names
        tiers = {"gpu": [], "cpu": [], "disk": [], "s3": []}
        for adapter_name in names:
            tier = self._get_local_tier(adapter_name)
            tiers.setdefault(tier, []).append(adapter_name)
        for tier in tiers:
            tiers[tier].sort()
        return tiers

    # def _reset_local_adapter_cache(self, adapter_names: list[str] | None = None) -> dict:
    #     target_adapters = list(adapter_names or self.lora_names)
    #     cleared_disk = []
    #     state_changes = []

    #     for adapter_name in target_adapters:
    #         old_tier = self._get_local_tier(adapter_name)

    #         self._local_gpu_lru.pop(adapter_name, None)
    #         self._local_cpu_lru.pop(adapter_name, None)

    #         adapter_path = self._get_local_adapter_path(adapter_name)
    #         if os.path.isdir(adapter_path):
    #             shutil.rmtree(adapter_path)
    #             cleared_disk.append(adapter_name)

    #         new_tier = self._get_initial_local_tier(adapter_name)
    #         if old_tier != new_tier:
    #             self._update_node_tier(adapter_name, self.my_ip, old_tier, new_tier)
    #             state_changes.append({
    #                 "adapter": adapter_name,
    #                 "old_tier": old_tier,
    #                 "new_tier": new_tier,
    #             })
    #         else:
    #             tiers = self._ensure_peer_tiers(adapter_name)
    #             for tier_name, nodes in tiers.items():
    #                 nodes.discard(self.my_ip)
    #             tiers[new_tier].add(self.my_ip)

    #     return {
    #         "node": self.my_ip,
    #         "cleared_disk": sorted(cleared_disk),
    #         "state_changes": state_changes,
    #         "local_adapters": self._get_local_tier_lists(target_adapters),
    #     }

    def _update_node_tier(self, adapter_name: str, node_ip: str,
                          old_tier: str, new_tier: str):
        """O(1) tier transition for any node in the cluster map."""
        tiers = self._ensure_peer_tiers(adapter_name)
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
            old_tier = self._get_local_tier(adapter_name)

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

    def _get_local_tier(self, adapter_name: str) -> str:
        """Return current memory tier for an adapter on this node."""
        if adapter_name in self._local_gpu_lru:
            return "gpu"
        if adapter_name in self._local_cpu_lru:
            return "cpu"
        if os.path.isdir(self._get_local_adapter_path(adapter_name)):
            return "disk"
        if USE_S3_ADAPTERS:
            return "s3"
        return "disk"
