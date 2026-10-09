"""A small Days at Three Branches starter built entirely from ``sandbox.village``."""
from collections import deque
from collections.abc import Mapping
from typing import cast

from sandbox.village import action, day, geometry, layout, me, people, props

_IDLE_EMOTES = (
    "nod",
    "shake_head",
    "point",
    "laugh",
    "shrug",
    "startle",
    "sleep",
    "sweep",
)
_MIDDAY_GATHER_TICK = 400
_MIDDAY_SWEEP_TICK = 600
_MIDDAY_PARTICIPANTS = 5
_MIDDAY_WINDOW_END = 720
_MIDDAY_RESIDENT_INDICES = (0, 1, 2, 4, 6)

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
    goals: set[tuple[int, int]],
) -> list[dict[str, int]]:
    """Find the shortest route from a cell to any walkable goal cell."""

    start_key = (start["x"], start["y"])
    queue = deque([start_key])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {
        start_key: None
    }
    reached: tuple[int, int] | None = None

    while queue:
        current = queue.popleft()
        if current in goals:
            reached = current
            break

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

    if reached is None:
        return []

    path: list[dict[str, int]] = []
    current = reached

    while current is not None:
        path.append({
            "x": current[0],
            "y": current[1],
        })
        current = previous[current]

    return list(reversed(path))


def _approach_cells(observation, target: Mapping[str, object]) -> set[tuple[int, int]]:
    """Find walkable cell centres where using selects this exact prop."""
    frame = layout.frame(observation)
    target_cell = cast(Mapping[str, int], target["cell"])
    target_id = target["id"]
    goals: set[tuple[int, int]] = set()

    for y in range(max(0, target_cell["y"] - 4), min(int(frame["cells_y"]), target_cell["y"] + 5)):
        for x in range(max(0, target_cell["x"] - 4), min(int(frame["cells_x"]), target_cell["x"] + 5)):
            cell = {"x": x, "y": y}
            if not layout.walkable(observation, cell):
                continue
            candidate_observation = dict(observation)
            candidate_self = dict(cast(Mapping[str, object], observation["self"]))
            candidate_self["position"] = _cell_centre(cell, float(frame["cell_size"]))
            candidate_observation["self"] = candidate_self
            usable = props.usable(candidate_observation)
            if usable is not None and usable["id"] == target_id:
                goals.add((x, y))

    return goals


def _lineup_cells(observation, count: int) -> list[tuple[int, int]]:
    """Choose one contiguous walkable row near the village spawn."""
    frame = layout.frame(observation)
    spawn = cast(Mapping[str, float], layout.spawn(observation))
    spawn_cell = layout.cell_at(observation, spawn)
    if spawn_cell is None:
        return []

    width, height = int(frame["cells_x"]), int(frame["cells_y"])
    spawn_y = spawn_cell["y"]
    for offset in range(max(spawn_y, height - spawn_y)):
        rows = {spawn_y - offset, spawn_y + offset}
        candidates: list[tuple[float, list[tuple[int, int]]]] = []
        for y in rows:
            if not 0 <= y < height:
                continue
            run: list[tuple[int, int]] = []
            runs: list[list[tuple[int, int]]] = []
            for x in range(width):
                if layout.walkable(observation, {"x": x, "y": y}):
                    run.append((x, y))
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)

            for walkable_run in runs:
                for start in range(len(walkable_run) - count + 1):
                    segment = walkable_run[start : start + count]
                    if any(
                        (cell[0] + 0.5 - float(spawn["x"])) ** 2
                        + (cell[1] + 0.5 - float(spawn["y"])) ** 2
                        < 2.25
                        for cell in segment
                    ):
                        continue
                    centre_x = (segment[0][0] + segment[-1][0]) / 2
                    distance = abs(centre_x - float(spawn["x"])) + abs(
                        (y + 0.5) - float(spawn["y"])
                    )
                    candidates.append((distance, segment))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]

    return []


def _home_cell(observation) -> tuple[int, int] | None:
    """Choose an interior cell inside this villager's home."""
    building = layout.building(observation, me.home(observation))
    if building is None:
        return None
    anchor = cast(Mapping[str, int], building["cell"])
    cells = []
    for y in range(anchor["y"], anchor["y"] + 8):
        for x in range(anchor["x"], anchor["x"] + 9):
            cell = {"x": x, "y": y}
            if layout.ground_at(observation, cell) == "interior" and layout.walkable(observation, cell):
                cells.append((x, y))
    if not cells:
        doorway = layout.doorway(observation, str(building["id"]))
        doorway_cell = None if doorway is None else layout.cell_at(observation, doorway)
        return None if doorway_cell is None else (doorway_cell["x"], doorway_cell["y"])

    centre = (anchor["x"] + 4, anchor["y"] + 3)
    return min(cells, key=lambda cell: abs(cell[0] - centre[0]) + abs(cell[1] - centre[1]))


class Agent:
    def reset(self, seed, observation) -> None:
        self._cell_size = float(
            layout.frame(observation)["cell_size"]
        )

        all_targets = [
            cast(Mapping[str, object], prop)
            for prop in props.all(observation)
        ]
        self._rng = me.rng(observation, seed)
        roster_ids = [
            str(person["id"])
            for person in people.roster(observation)
            if people.is_npc(str(person["id"]))
        ]
        own_id = me.player_id(observation)
        resident_index = roster_ids.index(own_id) if own_id in roster_ids else None
        participant_indices = (
            tuple(range(len(roster_ids)))
            if len(roster_ids) <= _MIDDAY_PARTICIPANTS
            else _MIDDAY_RESIDENT_INDICES
        )
        self._targets = [
            target
            for index, target in enumerate(all_targets)
            if resident_index is not None and index % len(roster_ids) == resident_index
        ]
        self._rng.shuffle(self._targets)
        self._target_index = 0
        self._route: list[dict[str, int]] = []
        self._route_goals: frozenset[tuple[int, int]] | None = None
        self._approaches: dict[str, set[tuple[int, int]]] = {}
        self._failed_approaches: dict[str, set[tuple[int, int]]] = {}
        self._pending_target: str | None = None
        self._blocked_use = False
        self._announced = False
        self._visitor_greeted = False
        self._lineup = _lineup_cells(
            observation,
            max(len(roster_ids), _MIDDAY_PARTICIPANTS * 2),
        )
        self._lineup_slot = (
            self._lineup[resident_index]
            if resident_index is not None
            and resident_index in participant_indices
            and self._lineup
            else None
        )
        self._midday_participant = resident_index in participant_indices
        self._midday_stage: str | None = None
        self._midday_sweeps = 2
        self._midday_done = False
        self._home_cell = _home_cell(observation) if resident_index is not None else None

    def act(self, observation):
        heading = me.heading(observation)
        seen_visitor = any(
            person["id"] == "player_0"
            for person in people.seen(observation)
        )
        expression = "none"
        if seen_visitor and not self._visitor_greeted:
            expression = "wave"
            self._visitor_greeted = True

        if self._pending_target is not None:
            current_expression = me.expression(observation)
            if (
                current_expression["type"] == "use"
                and current_expression["target"] == self._pending_target
            ):
                self._target_index += 1
                self._announced = False
                self._clear_route()
            else:
                self._blocked_use = True
            self._pending_target = None

        if (
            self._midday_participant
            and not self._midday_done
            and day.tick(observation) >= _MIDDAY_GATHER_TICK
        ):
            if self._midday_stage is None:
                self._midday_stage = "gather"
                self._clear_route()
            if self._lineup_slot is not None:
                move, reached = self._follow_route(
                    observation,
                    {self._lineup_slot},
                    expression,
                )
                if move is not None:
                    return move
                if not reached:
                    self._midday_done = True
                elif day.tick(observation) < _MIDDAY_SWEEP_TICK:
                    return action.stand(heading, expression)
                else:
                    self._midday_stage = "sweep"
            if self._midday_stage == "sweep":
                self._midday_sweeps -= 1
                if self._midday_sweeps == 0:
                    self._midday_done = True
                    self._midday_stage = None
                    self._clear_route()
                return action.stand(heading, "sweep")

        if self._blocked_use:
            self._blocked_use = False
            return action.stand(heading, self._idle_expression(observation))

        if self._target_index < len(self._targets):
            target = self._targets[self._target_index]
            target_id = str(target["id"])
            if target_id not in self._approaches:
                self._approaches[target_id] = _approach_cells(observation, target)
            goals = self._approaches[target_id]
            failed = self._failed_approaches.setdefault(target_id, set())
            goals = goals - failed
            if not goals:
                self._target_index += 1
                self._announced = False
                self._clear_route()
                return action.stand(heading, self._idle_expression(observation))

            move, reached = self._follow_route(observation, goals, expression)
            if move is not None:
                return move
            if not reached:
                self._target_index += 1
                self._announced = False
                self._clear_route()
                return action.stand(heading, self._idle_expression(observation))

            usable = props.usable(observation)
            if usable is not None and usable["id"] == target_id:
                self._pending_target = target_id
                return action.stand(heading, "use")

            here_cell = layout.cell_at(observation, me.position(observation))
            if here_cell is not None:
                failed.add((here_cell["x"], here_cell["y"]))
            self._clear_route()
            return action.stand(heading, self._idle_expression(observation))

        if self._home_cell is not None:
            move, reached = self._follow_route(
                observation,
                {self._home_cell},
                expression,
            )
            if move is not None:
                return move
            if reached:
                return action.stand(heading, "sleep")

        return action.stand(heading, "sleep")

    def _clear_route(self) -> None:
        self._route = []
        self._route_goals = None

    def _idle_expression(self, observation) -> str:
        emotes = _IDLE_EMOTES
        if (
            _MIDDAY_GATHER_TICK <= day.tick(observation) <= _MIDDAY_WINDOW_END
        ):
            emotes = _IDLE_EMOTES[:-1]
        return self._rng.choice(emotes)

    def _follow_route(self, observation, goals, expression):
        here = me.position(observation)
        here_cell = layout.cell_at(observation, here)
        if here_cell is None:
            return action.stand(me.heading(observation), expression), False
        here_key = (here_cell["x"], here_cell["y"])
        goal_key = frozenset(goals)
        if self._route_goals != goal_key or (
            self._route and here_key not in {
                (cell["x"], cell["y"]) for cell in self._route
            }
        ):
            self._route = _route(observation, here_cell, set(goal_key))
            self._route_goals = goal_key
        if not self._route:
            return None, False

        route_index = next(
            (index for index, cell in enumerate(self._route)
             if cell["x"] == here_key[0] and cell["y"] == here_key[1]),
            None,
        )
        if route_index is None:
            self._clear_route()
            return None, False
        if route_index + 1 == len(self._route):
            return None, True

        next_cell = self._route[route_index + 1]
        target = _cell_centre(next_cell, self._cell_size)
        return action.walk(geometry.heading_to(here, target), 1.0, expression), False

    def chat(self, inbox: list[dict]) -> list[dict]:
        """Tell nearby villagers which independent task this instance owns."""

        if self._target_index >= len(self._targets) or self._announced:
            return []
        target = self._targets[self._target_index]
        self._announced = True
        return [{"to": None, "text": f"I am working on {target['id']}."}]
