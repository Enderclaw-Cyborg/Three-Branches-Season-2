"""A social scripted visitor for Days at Three Branches."""

from __future__ import annotations

import math
import random
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, cast

from sandbox.village import layout, props


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _position(record: Mapping[str, Any]) -> tuple[float, float]:
    point = record.get("position")
    if not isinstance(point, Mapping):
        return 0.0, 0.0
    return _number(point.get("x")), _number(point.get("y"))


def _heading_to(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360.0


def _cell_centre(cell: tuple[int, int], cell_size: float) -> tuple[float, float]:
    return (cell[0] + 0.5) * cell_size, (cell[1] + 0.5) * cell_size


def _route_to_village(observation: object) -> list[tuple[int, int]]:
    """Find the shortest route from the visitor to the reachable cell nearest the map centre."""
    state = cast(Mapping[str, object], observation)
    frame = layout.frame(state)
    record = _self_record(observation)
    position = record.get("position")
    if not isinstance(position, Mapping):
        return []
    start = layout.cell_at(state, cast(Mapping[str, object], position))
    if start is None:
        return []

    width, height = int(frame["cells_x"]), int(frame["cells_y"])
    centre = (width // 2, height // 2)
    start_key = (start["x"], start["y"])
    queue = deque([start_key])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
    best = start_key
    best_distance = abs(best[0] - centre[0]) + abs(best[1] - centre[1])

    while queue:
        current = queue.popleft()
        distance = abs(current[0] - centre[0]) + abs(current[1] - centre[1])
        if distance < best_distance:
            best, best_distance = current, distance
        if distance == 0:
            best = current
            break

        for neighbor in (
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        ):
            if neighbor in previous or not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if layout.can_step(
                state,
                {"x": current[0], "y": current[1]},
                {"x": neighbor[0], "y": neighbor[1]},
            ):
                previous[neighbor] = current
                queue.append(neighbor)

    path = []
    current: tuple[int, int] | None = best
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _route_to_nearest_bench(
    observation: object,
) -> tuple[str, list[tuple[int, int]]] | None:
    """Find the shortest legal route to a cell that can use the nearest bench."""
    state = cast(Mapping[str, object], observation)
    frame = layout.frame(state)
    position = _position(_self_record(observation))
    start = layout.cell_at(state, {"x": position[0], "y": position[1]})
    if start is None:
        return None

    width, height = int(frame["cells_x"]), int(frame["cells_y"])
    start_key = (start["x"], start["y"])
    queue = deque([start_key])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
    while queue:
        current = queue.popleft()
        for neighbor in (
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        ):
            if neighbor in previous or not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if layout.can_step(
                state,
                {"x": current[0], "y": current[1]},
                {"x": neighbor[0], "y": neighbor[1]},
            ):
                previous[neighbor] = current
                queue.append(neighbor)

    candidates: list[tuple[int, str, list[tuple[int, int]]]] = []
    for bench in props.all(state):
        if bench["type"] != "bench":
            continue
        bench_cell = cast(Mapping[str, int], bench["cell"])
        for y in range(max(0, bench_cell["y"] - 4), min(height, bench_cell["y"] + 5)):
            for x in range(max(0, bench_cell["x"] - 4), min(width, bench_cell["x"] + 5)):
                goal = (x, y)
                if goal not in previous or not layout.walkable(state, {"x": x, "y": y}):
                    continue
                candidate = dict(state)
                candidate_self = dict(_self_record(observation))
                cell_size = float(frame["cell_size"])
                candidate_self["position"] = {
                    "x": (x + 0.5) * cell_size,
                    "y": (y + 0.5) * cell_size,
                }
                candidate["self"] = candidate_self
                usable = props.usable(candidate)
                if usable is None or usable["id"] != bench["id"]:
                    continue

                path: list[tuple[int, int]] = []
                current: tuple[int, int] | None = goal
                while current is not None:
                    path.append(current)
                    current = previous[current]
                candidates.append((len(path), str(bench["id"]), list(reversed(path))))

    if not candidates:
        return None
    _, bench_id, path = min(candidates, key=lambda item: (item[0], item[1]))
    return bench_id, path


_NPC_PLAYER_ID = re.compile(r"player_[1-9][0-9]*\Z")


def _self_record(observation: object) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        record = observation.get("self")
        if isinstance(record, Mapping):
            return record
    return {}


def _seen_people(observation: object) -> list[Mapping[str, Any]]:
    if not isinstance(observation, Mapping):
        return []
    seen = observation.get("seen")
    if not isinstance(seen, Sequence) or isinstance(seen, str | bytes):
        return []
    return [person for person in seen if isinstance(person, Mapping) and isinstance(person.get("id"), str)]


class Agent:
    """Wander, greet a seen villager, linger, then continue through the village."""

    def reset(self, seed: object, observation: object) -> None:
        """Start a fresh visit. Builtins deliberately do not use the session seed."""
        del seed
        self._rng = random.Random()
        self._heading = self._rng.uniform(0.0, 360.0)
        self._remaining = self._rng.randint(18, 36)
        bench_route = _route_to_nearest_bench(observation)
        self._bench_target = None if bench_route is None else bench_route[0]
        self._bench_route = [] if bench_route is None else bench_route[1]
        self._bench_sit_remaining = 20
        self._mode = "bench" if bench_route is not None else "enter"
        self._village_route = [] if bench_route is not None else _route_to_village(observation)
        self._target: str | None = None
        self._linger = 0
        self._greeting_due = False
        self._replied = False

    def act(self, observation: object) -> dict[str, float | int]:
        """Choose a social movement and expression from the current observation."""
        me = _self_record(observation)
        position = _position(me)
        seen = _seen_people(observation)
        target = self._find_target(seen, position)

        if self._mode == "bench":
            state = cast(Mapping[str, object], observation)
            expression = me_record = _self_record(observation).get("expression")
            if (
                isinstance(me_record, Mapping)
                and me_record.get("type") == "use"
                and me_record.get("target") == self._bench_target
            ):
                self._bench_sit_remaining -= 1
            if self._bench_sit_remaining <= 0:
                self._mode = "wander"
            else:
                cell = layout.cell_at(state, {"x": position[0], "y": position[1]})
                route_cells = self._bench_route
                if cell is not None and route_cells:
                    cell_key = (cell["x"], cell["y"])
                    if cell_key not in route_cells:
                        bench_route = _route_to_nearest_bench(observation)
                        if bench_route is not None:
                            self._bench_target, self._bench_route = bench_route
                            route_cells = self._bench_route
                    route_index = next(
                        (index for index, route_cell in enumerate(route_cells) if route_cell == cell_key),
                        None,
                    )
                    if route_index is not None and route_index + 1 < len(route_cells):
                        size = float(layout.frame(state)["cell_size"])
                        destination = _cell_centre(route_cells[route_index + 1], size)
                        self._heading = _heading_to(position, destination)
                        return {"heading": self._heading, "speed": 0.8, "action": 0}

                    usable = props.usable(state)
                    if usable is not None and usable["id"] == self._bench_target:
                        return {"heading": self._heading, "speed": 0.0, "action": 1}
                    bench_route = _route_to_nearest_bench(observation)
                    if bench_route is not None:
                        self._bench_target, self._bench_route = bench_route
                        return {"heading": self._heading, "speed": 0.0, "action": 0}
                self._mode = "wander"

        if self._mode == "enter":
            cell = layout.cell_at(
                cast(Mapping[str, object], observation),
                {"x": position[0], "y": position[1]},
            )
            route_cells = self._village_route
            if cell is not None and route_cells:
                cell_key = (cell["x"], cell["y"])
                if cell_key not in route_cells:
                    self._village_route = _route_to_village(observation)
                    route_cells = self._village_route
                route_index = next(
                    (index for index, route_cell in enumerate(route_cells) if route_cell == cell_key),
                    None,
                )
                if route_index is not None and route_index + 1 < len(route_cells):
                    size = float(layout.frame(cast(Mapping[str, object], observation))["cell_size"])
                    destination = _cell_centre(route_cells[route_index + 1], size)
                    self._heading = _heading_to(position, destination)
                    return {"heading": self._heading, "speed": 0.8, "action": 0}
            self._mode = "wander"

        if self._mode == "wander" and target is not None:
            self._target = str(target["id"])
            self._mode = "approach"
            self._replied = False

        if self._mode == "approach":
            if target is None:
                self._start_wandering()
            else:
                self._heading = _heading_to(position, _position(target))
                distance = math.dist(position, _position(target))
                if distance > 1.7:
                    return {"heading": self._heading, "speed": 0.8, "action": 0}
                self._mode = "linger"
                self._linger = self._rng.randint(4, 7)
                self._greeting_due = True
                return {"heading": self._heading, "speed": 0.0, "action": 2}

        if self._mode == "linger":
            self._linger -= 1
            if self._linger <= 0:
                self._mode = "move_on"
                self._remaining = self._rng.randint(12, 22)
                self._heading = (self._heading + self._rng.choice((135.0, 180.0, 225.0))) % 360.0
            return {"heading": self._heading, "speed": 0.0, "action": 0}

        if self._mode == "move_on":
            self._remaining -= 1
            if self._remaining <= 0:
                self._start_wandering()
            return {"heading": self._heading, "speed": 0.75, "action": 0}

        self._remaining -= 1
        if self._remaining <= 0:
            self._heading = (self._heading + self._rng.choice((-90.0, -60.0, 60.0, 90.0))) % 360.0
            self._remaining = self._rng.randint(18, 36)
        return {"heading": self._heading, "speed": 0.7, "action": 0}

    def chat(self, inbox: object) -> list[dict[str, str]]:
        """Offer one canned greeting, then one short reply to the same villager."""
        if self._target is None:
            return []
        if self._greeting_due:
            self._greeting_due = False
            return [{"to": self._target, "text": "Hello. How is your day going?"}]
        if self._replied or not isinstance(inbox, Sequence) or isinstance(inbox, str | bytes):
            return []
        for message in inbox:
            if isinstance(message, Mapping) and message.get("from") == self._target:
                self._replied = True
                return [{"to": self._target, "text": "Thank you. I am glad to be here."}]
        return []

    def _find_target(
        self, seen: list[Mapping[str, Any]], position: tuple[float, float]
    ) -> Mapping[str, Any] | None:
        """Keep pursuing one villager, or choose the nearest newly seen villager."""
        candidates = [person for person in seen if _NPC_PLAYER_ID.fullmatch(str(person.get("id", "")))]
        if not candidates:
            return None
        for person in candidates:
            if person.get("id") == self._target:
                return person
        return min(candidates, key=lambda person: math.dist(position, _position(person)))

    def _start_wandering(self) -> None:
        self._mode = "wander"
        self._target = None
        self._heading = (self._heading + self._rng.choice((-105.0, -75.0, 75.0, 105.0))) % 360.0
        self._remaining = self._rng.randint(18, 36)
