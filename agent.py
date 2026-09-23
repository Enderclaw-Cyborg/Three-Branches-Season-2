"""A small Days at Three Branches starter built entirely from ``sandbox.village``."""
from collections import deque
from collections.abc import Mapping
from typing import cast

from sandbox.village import action, geometry, layout, me, people, props

def _cell_centre(
    cell: Mapping[str, int],
    cell_size: float,
) -> dict[str, float]:
    return {
        "x": (cell["x"] + 0.5) * cell_size,
        "y": (cell["y"] + 0.5) * cell_size,
    }


def _route(
    observation,
    start: Mapping[str, int],
    goal: Mapping[str, int],
) -> list[dict[str, int]]:
    """Find a route to the nearest walkable cell beside the target."""

    start_key = (start["x"], start["y"])
    goal_key = (goal["x"], goal["y"])

    queue = deque([start_key])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {
        start_key: None
    }

    best = start_key

    while queue:
        current = queue.popleft()

        current_distance = (
            abs(current[0] - goal_key[0])
            + abs(current[1] - goal_key[1])
        )
        best_distance = (
            abs(best[0] - goal_key[0])
            + abs(best[1] - goal_key[1])
        )

        if current_distance < best_distance:
            best = current

        neighbors = (
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
        )

        for neighbor in neighbors:
            if neighbor in previous:
                continue

            current_cell = {
                "x": current[0],
                "y": current[1],
            }
            neighbor_cell = {
                "x": neighbor[0],
                "y": neighbor[1],
            }

            if layout.can_step(
                observation,
                current_cell,
                neighbor_cell,
            ):
                previous[neighbor] = current
                queue.append(neighbor)

    if best not in previous:
        return []

    path: list[dict[str, int]] = []
    current = best

    while current is not None:
        path.append({
            "x": current[0],
            "y": current[1],
        })
        current = previous[current]

    return list(reversed(path))


class Agent:
    def reset(self, seed, observation) -> None:
        self._cell_size = float(
            layout.frame(observation)["cell_size"]
        )

        self._targets = [
            cast(Mapping[str, object], prop)
            for prop in props.all(observation)
        ]
        self._rng = me.rng(observation, seed)
        self._rng.shuffle(self._targets)
        self._target_index = 0
        self._pause_remaining = 0
        self._route: list[dict[str, int]] = []
        self._announced = False

    def act(self, observation):
        heading = me.heading(observation)
        expression = "wave" if people.seen(observation) else "none"

        if not self._targets:
            return action.stand(heading, expression)

        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            return action.stand(heading, expression)

        if self._target_index >= len(self._targets):
            self._rng.shuffle(self._targets)
            self._target_index = 0

        target = self._targets[self._target_index]
        target_id = target["id"]

        usable = props.usable(observation)

        if (
            usable is not None
            and usable["type"] == target["type"]
            and usable["id"] == target_id
        ):
            self._target_index += 1
            self._route = []
            self._pause_remaining = self._rng.randint(2, 8)
            self._announced = False
            return action.stand(heading, "use")

        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)

        if here_cell is not None and not self._route:
            target_cell = cast(
                Mapping[str, int],
                target["cell"],
            )
            self._route = _route(
                observation,
                here_cell,
                target_cell,
            )

        if here_cell is not None and self._route:
            try:
                route_index = next(
                    index
                    for index, cell in enumerate(self._route)
                    if cell == here_cell
                )
            except StopIteration:
                route_index = len(self._route) - 1

            if route_index + 1 < len(self._route):
                next_cell = self._route[route_index + 1]
                target = _cell_centre(
                    next_cell,
                    self._cell_size,
                )
                return action.walk(
                    geometry.heading_to(here, target),
                    1.0,
                    expression,
                )

        return action.stand(heading, expression)

    def chat(self, inbox: list[dict]) -> list[dict]:
        """Tell nearby villagers which independent task this instance owns."""

        if not self._targets or self._announced:
            return []
        target = self._targets[self._target_index % len(self._targets)]
        self._announced = True
        return [{"to": None, "text": f"I am working on {target['id']}."}]
