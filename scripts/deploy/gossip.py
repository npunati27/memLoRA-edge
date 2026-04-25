import asyncio
import time

from .config import SERVE_PORT, logger


class GossipMixin:
    """Peer gossip: queue-length broadcasting, adapter-state propagation, RTT queries."""

    def _start_gossip_loop(self):
        """Start the background gossip loop. Safe to call from __init__ after engine ready."""
        if self._gossip_task is None:
            try:
                loop = asyncio.get_event_loop()
                self._gossip_task = loop.create_task(self._gossip_queue_loop())
                self._gossip_running = True
                logger.info(f"[gossip] Started gossip loop for {self.my_ip}")
            except RuntimeError:
                self._gossip_running = False
                logger.warning(
                    "[gossip] No event loop available, gossip will start on first request"
                )

    async def _ensure_session(self):
        """Lazily create and return the shared aiohttp session."""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            import aiohttp
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=2)
            )
        return self._aiohttp_session

    # ── Queue length gossip ───────────────────────────────────────────────

    async def _gossip_queue_loop(self):
        """Background task that broadcasts local queue length to all peers every 150ms."""
        await asyncio.sleep(1)
        peer_count = len([p for p in self.peer_ips if p != self.my_ip])
        logger.info(f"[gossip] Gossip loop active, broadcasting to {peer_count} peers")

        while self._gossip_running:
            try:
                await self._broadcast_queue_length()
            except Exception as e:
                logger.error(f"[gossip] Broadcast error: {e}")
            await asyncio.sleep(0.15)

    async def _broadcast_queue_length(self):
        """Send current queue length to all peers."""
        msg = {
            "type": "queue_length",
            "node": self.my_ip,
            "queue_len": self._ongoing,
            "ts": time.time(),
        }
        await self._broadcast_to_peers(msg)

    def _handle_queue_gossip(self, body: dict):
        """Process incoming queue length gossip from a peer."""
        node = body.get("node")
        queue_len = body.get("queue_len", 0)
        ts = body.get("ts", 0)

        if node and node != self.my_ip and node in self._peer_queue_lengths:
            if ts > self._peer_queue_timestamps.get(node, 0):
                self._peer_queue_lengths[node] = queue_len
                self._peer_queue_timestamps[node] = ts

    # ── Broadcast helpers ─────────────────────────────────────────────────

    async def _broadcast_to_peers(self, msg: dict):
        """Send a message to all peers concurrently."""
        tasks = []
        for peer in self.peer_ips:
            if peer != self.my_ip:
                tasks.append(self._send_gossip(peer, msg))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_gossip(self, peer_ip: str, msg: dict):
        """Send a gossip message to a single peer."""
        url = f"http://{peer_ip}:{SERVE_PORT}/internal/gossip"
        try:
            session = await self._ensure_session()
            async with session.post(url, json=msg) as resp:
                await resp.read()
        except Exception:
            pass

    # ── Adapter state gossip ──────────────────────────────────────────────

    async def _broadcast_state_change(self, adapter_name: str,
                                      old_tier: str, new_tier: str):
        """Broadcast an adapter tier change to all peers immediately."""
        msg = {
            "type": "adapter_state",
            "node": self.my_ip,
            "adapter": adapter_name,
            "old_tier": old_tier,
            "new_tier": new_tier,
            "ts": time.time(),
        }
        await self._broadcast_to_peers(msg)

    def _handle_adapter_state_gossip(self, body: dict):
        """Process incoming adapter state change from a peer."""
        node = body.get("node")
        adapter = body.get("adapter")
        new_tier = body.get("new_tier")
        ts = body.get("ts", 0)

        if not all([node, adapter, new_tier]) or node == self.my_ip:
            return

        key = (adapter, node)
        if ts <= self._adapter_state_timestamps.get(key, 0):
            return
        self._adapter_state_timestamps[key] = ts

        if adapter not in self._peer_adapter_state:
            self._peer_adapter_state[adapter] = {
                "gpu": set(), "cpu": set(), "disk": set(), "s3": set(),
            }

        tiers = self._peer_adapter_state[adapter]
        for tier_set in tiers.values():
            tier_set.discard(node)
        if new_tier in tiers:
            tiers[new_tier].add(node)

    # ── RTT-based queue queries ───────────────────────────────────────────

    async def _query_peer_queue(self, peer_ip: str) -> int:
        """Query a peer's queue length directly via HTTP (RTT approach)."""
        import aiohttp
        url = f"http://{peer_ip}:{SERVE_PORT}/internal/queue"
        try:
            session = await self._ensure_session()
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=1)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    queue_len = data.get("queue_len", 0)
                    ts = data.get("ts", time.time())
                    self._peer_queue_lengths[peer_ip] = queue_len
                    self._peer_queue_timestamps[peer_ip] = ts
                    return queue_len
        except Exception:
            pass
        return self._peer_queue_lengths.get(peer_ip, 0)

    async def _query_all_peer_queues(self) -> dict[str, int]:
        """Query all peers' queue lengths concurrently."""
        tasks = {}
        for peer in self.peer_ips:
            if peer != self.my_ip:
                tasks[peer] = self._query_peer_queue(peer)
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            return {
                ip: (r if isinstance(r, int) else 0)
                for ip, r in zip(tasks.keys(), results)
            }
        return {}
