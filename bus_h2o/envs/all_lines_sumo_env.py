"""Canonical all-lines adapter for online rollouts in SUMO."""

from __future__ import annotations

from .sumo_gym_env import SumoGymEnv


class AllLinesSumoEnv(SumoGymEnv):
    """Expose all SUMO decision events without legacy snapshot side effects.

    ``SumoGymEnv`` already emits every active line and uses the same stable,
    alphabetically sorted line index as the v3 dataset.  The canonical
    protocol starts every episode from time zero and therefore disables its
    legacy state-file pool, which otherwise performs synchronous disk I/O
    during online collection.
    """

    def __init__(
        self,
        sumo_dir: str,
        edge_xml: str,
        schedule_xml: str | None = None,
        *,
        episode_seconds: int = 24_000,
        sumo_seed: int = 42,
        gui: bool = False,
        sumo_cfg: str | None = None,
        passenger_obj_xml: str | None = None,
        observation_schema: str = "v3",
        sumo_backend: str = "libsumo",
    ) -> None:
        if episode_seconds <= 0:
            raise ValueError("episode_seconds must be positive")
        if gui:
            raise ValueError("the canonical all-lines protocol is headless-only")
        super().__init__(
            sumo_dir=sumo_dir,
            edge_xml=edge_xml,
            schedule_xml=schedule_xml,
            max_steps=int(episode_seconds),
            gui=gui,
            # This legacy parent argument selects only a reference edge map for
            # optional snapshot visualization. SumoRLBridge does not use it to
            # filter decision events; every scheduled service remains active.
            line_id="7X",
            seed=sumo_seed,
            sumo_cfg=sumo_cfg,
            passenger_obj_xml=passenger_obj_xml,
            observation_schema=observation_schema,
            sumo_backend=sumo_backend,
        )
        self.episode_seconds = int(episode_seconds)
        self.sumo_seed = int(sumo_seed)
        self._snapshot_pool_max = 0

    def reset(self, snapshot=None):
        if snapshot is not None:
            raise NotImplementedError(
                "snapshot reset is intentionally excluded from the canonical v3 protocol"
            )
        return super().reset(snapshot=None)

    def close(self) -> None:
        if self._bridge is None:
            return
        close = getattr(self._bridge, "shutdown", None)
        if callable(close):
            close()
