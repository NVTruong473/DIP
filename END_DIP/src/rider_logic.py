from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.common import Detection
from src.utils.geometry import expand_box, head_region, point_in_box


@dataclass
class RiderObservation:
    person: Detection
    vehicle: Detection
    helmet: Optional[Detection]
    instantaneous_status: str
    stable_status: str = "UNKNOWN"

    @property
    def track_id(self) -> Optional[int]:
        return self.person.track_id if self.person.track_id is not None else self.vehicle.track_id


def associate_riders(
    scene_detections: List[Detection],
    helmet_detections: List[Detection],
    frame_shape,
) -> List[RiderObservation]:
    persons = [d for d in scene_detections if d.class_id == 0]
    vehicles = [d for d in scene_detections if d.class_id in (1, 3)]
    riders: List[RiderObservation] = []

    for person in persons:
        best_vehicle = None
        best_score = float("inf")
        px, py = person.bottom_center

        for vehicle in vehicles:
            # The rider sits above the bike; expanding upward is substantially
            # more reliable than raw IoU for two differently shaped boxes.
            region = expand_box(vehicle.box, frame_shape, x_scale=2.0, y_scale=3.2, upward_bias=0.32)
            if not point_in_box((px, py), region):
                continue
            vx, vy = vehicle.center
            score = abs(px - vx) + 0.35 * abs(py - vy)
            if score < best_score:
                best_score = score
                best_vehicle = vehicle

        if best_vehicle is None:
            continue

        hbox = head_region(person.box, frame_shape)
        candidates = [h for h in helmet_detections if point_in_box(h.center, hbox)]
        helmet = max(candidates, key=lambda d: d.confidence) if candidates else None

        # Critical safety rule: a missed detection is UNKNOWN, not NO_HELMET.
        status = helmet.label if helmet is not None else "UNKNOWN"
        riders.append(RiderObservation(person, best_vehicle, helmet, status))

    return riders
