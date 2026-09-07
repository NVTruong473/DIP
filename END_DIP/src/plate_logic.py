from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.common import Detection


@dataclass
class VehiclePlate:
    vehicle: Detection
    plate: Detection


def _inside_expanded_vehicle(plate: Detection, vehicle: Detection, expand: float = 0.05) -> bool:
    vx1, vy1, vx2, vy2 = vehicle.box
    vw = max(1, vx2 - vx1)
    vh = max(1, vy2 - vy1)
    px, py = plate.center

    ex1 = vx1 - vw * expand
    ey1 = vy1 - vh * expand
    ex2 = vx2 + vw * expand
    ey2 = vy2 + vh * expand
    return ex1 <= px <= ex2 and ey1 <= py <= ey2


def _plate_is_plausible_for_vehicle(plate: Detection, vehicle: Detection) -> bool:
    vx1, vy1, vx2, vy2 = vehicle.box
    px1, py1, px2, py2 = plate.box
    vw = max(1, vx2 - vx1)
    vh = max(1, vy2 - vy1)
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)

    # License plates are small horizontal/near-square regions in the lower
    # portion of a car/bus/truck. Keep these limits intentionally permissive.
    width_ratio = pw / vw
    area_ratio = (pw * ph) / float(vw * vh)
    _, pcy = plate.center
    rel_y = (pcy - vy1) / vh

    if not (0.08 <= width_ratio <= 0.95):
        return False
    if not (0.001 <= area_ratio <= 0.20):
        return False
    if rel_y < 0.38:
        return False
    return True


def associate_car_plates(vehicles: List[Detection], plates: List[Detection]) -> List[VehiclePlate]:
    """Associate each plate with one car/bus/truck detection.

    Vehicle boxes are used only for filtering and tracking; callers can choose
    not to draw them, which keeps the final video visually clean.
    """
    pairs: List[VehiclePlate] = []
    used_plate_ids = set()

    for vehicle in sorted(vehicles, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])):
        vx1, vy1, vx2, vy2 = vehicle.box
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        vbottom_x = (vx1 + vx2) / 2.0
        vbottom_y = float(vy2)

        best_idx: Optional[int] = None
        best_score = float("inf")
        for i, plate in enumerate(plates):
            if i in used_plate_ids:
                continue
            if not _inside_expanded_vehicle(plate, vehicle):
                continue
            if not _plate_is_plausible_for_vehicle(plate, vehicle):
                continue

            pcx, pcy = plate.center
            # Prefer a plate near the vehicle's lower-center. Normalize by
            # vehicle dimensions so the score is comparable across distances.
            score = abs(pcx - vbottom_x) / vw + 0.35 * abs(pcy - vbottom_y) / vh
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            used_plate_ids.add(best_idx)
            plate = plates[best_idx]
            plate.track_id = vehicle.track_id
            plate.extra = {
                **(plate.extra or {}),
                "vehicle_class": vehicle.label,
                "vehicle_track_id": vehicle.track_id,
            }
            pairs.append(VehiclePlate(vehicle=vehicle, plate=plate))

    return pairs
