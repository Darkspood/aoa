from __future__ import annotations

from dataclasses import dataclass
from math import dist
from time import perf_counter


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class KMedianInstance:
    facilities: list[Point]
    clients: list[Point]

    def distance_matrix(self) -> list[list[float]]:
        return [
            [dist((facility.x, facility.y), (client.x, client.y)) for client in self.clients]
            for facility in self.facilities
        ]


@dataclass
class SolutionMetrics:
    algorithm: str
    open_facilities: list[int]
    assignment: list[int]
    cost: float
    runtime_seconds: float
    swap_iterations: int = 0
    improving_swaps: int = 0
    evaluated_swaps: int = 0


def _validate_k(k: int, num_facilities: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > num_facilities:
        raise ValueError("k cannot exceed the number of facilities.")


def total_cost(open_facilities: list[int], distances: list[list[float]]) -> tuple[float, list[int]]:
    assignment: list[int] = []
    total = 0.0
    for client_index in range(len(distances[0])):
        # Each client is assigned to its nearest currently open facility.
        best_facility = min(open_facilities, key=lambda facility: distances[facility][client_index])
        assignment.append(best_facility)
        total += distances[best_facility][client_index]
    return total, assignment


def _compute_closest_data(
    open_facilities: list[int], distances: list[list[float]]
) -> tuple[float, list[int], list[float], list[float]]:
    assignment: list[int] = []
    closest_distances: list[float] = []
    second_closest_distances: list[float] = []
    total = 0.0

    for client_index in range(len(distances[0])):
        # Local search needs both the closest and second-closest open facility so
        # it can estimate the effect of removing one facility during a swap.
        ranked = sorted(
            ((distances[facility][client_index], facility) for facility in open_facilities),
            key=lambda item: item[0],
        )
        best_distance, best_facility = ranked[0]
        second_best_distance = ranked[1][0] if len(ranked) > 1 else ranked[0][0]
        assignment.append(best_facility)
        closest_distances.append(best_distance)
        second_closest_distances.append(second_best_distance)
        total += best_distance

    return total, assignment, closest_distances, second_closest_distances


def greedy_kmedian(distances: list[list[float]], k: int) -> SolutionMetrics:
    _validate_k(k, len(distances))

    start = perf_counter()
    open_facilities: list[int] = []
    open_set: set[int] = set()
    assignment: list[int] = []
    current_cost = float("inf")
    facility_indices = range(len(distances))

    for _ in range(k):
        best_candidate = None
        best_cost = float("inf")
        best_assignment: list[int] = []
        for candidate in facility_indices:
            if candidate in open_set:
                continue
            # Evaluate the objective after opening one additional facility.
            candidate_set = open_facilities + [candidate]
            candidate_cost, candidate_assignment = total_cost(candidate_set, distances)
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_candidate = candidate
                best_assignment = candidate_assignment

        if best_candidate is None:
            raise RuntimeError("Unable to select a greedy facility.")
        open_facilities.append(best_candidate)
        open_set.add(best_candidate)
        current_cost = best_cost
        assignment = best_assignment

    runtime = perf_counter() - start
    return SolutionMetrics(
        algorithm="greedy",
        open_facilities=sorted(open_facilities),
        assignment=assignment,
        cost=current_cost,
        runtime_seconds=runtime,
    )


def local_search_kmedian(
    distances: list[list[float]],
    k: int,
    initial_open: list[int] | None = None,
    tolerance: float = 1e-9,
) -> SolutionMetrics:
    _validate_k(k, len(distances))

    if initial_open is None:
        initial_solution = greedy_kmedian(distances, k)
        open_facilities = initial_solution.open_facilities[:]
    else:
        if len(initial_open) != k:
            raise ValueError("The initial solution must open exactly k facilities.")
        open_facilities = sorted(initial_open)

    start = perf_counter()
    current_cost, assignment, closest_distances, second_closest_distances = _compute_closest_data(
        open_facilities, distances
    )
    improving_swaps = 0
    evaluated_swaps = 0
    passes = 0
    open_set = set(open_facilities)

    while True:
        passes += 1
        best_swap: tuple[int, int] | None = None
        best_cost = current_cost
        closed_facilities = [index for index in range(len(distances)) if index not in open_set]

        for open_facility in open_facilities:
            for closed_facility in closed_facilities:
                candidate_cost = 0.0
                for client_index, assigned_facility in enumerate(assignment):
                    incoming_distance = distances[closed_facility][client_index]
                    # If the outgoing facility currently serves this client, the client
                    # may need to fall back to its second-best open facility.
                    if assigned_facility == open_facility:
                        candidate_cost += min(second_closest_distances[client_index], incoming_distance)
                    else:
                        candidate_cost += min(closest_distances[client_index], incoming_distance)

                    # Stop early once this swap is already worse than the best swap seen.
                    if candidate_cost >= best_cost - tolerance:
                        break

                evaluated_swaps += 1
                if candidate_cost + tolerance < best_cost:
                    best_cost = candidate_cost
                    best_swap = (open_facility, closed_facility)

        if best_swap is None:
            break

        outgoing, incoming = best_swap
        open_facilities.remove(outgoing)
        open_facilities.append(incoming)
        open_facilities.sort()
        open_set.remove(outgoing)
        open_set.add(incoming)
        # Refresh exact assignments after committing the best improving swap.
        current_cost, assignment, closest_distances, second_closest_distances = _compute_closest_data(
            open_facilities, distances
        )
        improving_swaps += 1

    runtime = perf_counter() - start
    return SolutionMetrics(
        algorithm="local_search",
        open_facilities=open_facilities,
        assignment=assignment,
        cost=current_cost,
        runtime_seconds=runtime,
        swap_iterations=passes,
        improving_swaps=improving_swaps,
        evaluated_swaps=evaluated_swaps,
    )
