import asyncio
import time
import aiohttp
from .config import SERVE_PORT, logger

class ProbeMixin:
    
    def _start_probe_loop(self):
        self._probe_task = asyncio.create_task(self._probe_rtt_loop())
        logger.info("[probe] RTT probe loop started")

    async def _stop_probe_loop(self):
        if hasattr(self, "_probe_task") and not self._probe_task.done():
            self._probe_task.cancel()
            try:
                await self._probe_task
            except asyncio.CancelledError:
                pass

    async def _probe_rtt_loop(self):
        while True:
            for peer_ip in self.peer_ips:
                if peer_ip == self.my_ip:
                    continue
                await self._probe_peer(peer_ip)
            await asyncio.sleep(5) 

    async def _probe_peer(self, peer_ip: str):
        url = f"http://{peer_ip}:{SERVE_PORT}/internal/ping"
        try:
            session = await self._ensure_session()
            t0 = time.perf_counter()
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                await resp.json()
            rtt_ms = (time.perf_counter() - t0) * 1000

            # EWMA: smooth noise but track real degradation
            # 0.8 weight on history, 0.2 on new measurement
            prev = self._measured_rtt.get(peer_ip, rtt_ms)
            self._measured_rtt[peer_ip] = 0.8 * prev + 0.2 * rtt_ms

            # reset failure count on success
            self._probe_failures[peer_ip] = 0

            logger.debug(
                f"[probe] {peer_ip} RTT={rtt_ms:.1f}ms "
                f"EWMA={self._measured_rtt[peer_ip]:.1f}ms"
            )

        except Exception as e:
            # increment failure count
            self._probe_failures[peer_ip] = (
                self._probe_failures.get(peer_ip, 0) + 1
            )
            failures = self._probe_failures[peer_ip]

            if failures >= 3:
                # link is dead, exclude from routing
                self._measured_rtt[peer_ip] = float("inf")
                logger.warning(
                    f"[probe] {peer_ip} unreachable after {failures} attempts "
                    f"— marking dead"
                )
            else:
                logger.warning(
                    f"[probe] {peer_ip} probe failed ({failures}/3): {e}"
                )