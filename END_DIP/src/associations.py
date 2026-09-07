from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.common import Detection


@dataclass
class RiderMatch:
    person: Detection
    motorcycle: Detection
    helmet: Optional[Detection]

    @property
    def track_id(self):
        return self.motorcycle.track_id if self.motorcycle.track_id is not None else self.person.track_id


@dataclass
class PlateMatch:
    vehicle: Detection
    plate: Detection

    @property
    def track_id(self):
        return self.vehicle.track_id


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _inside(point, box, margin_x=0.0, margin_y=0.0):
    x, y = point
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return x1 - margin_x * w <= x <= x2 + margin_x * w and y1 - margin_y * h <= y <= y2 + margin_y * h


def associate_riders(scene: List[Detection], helmets: List[Detection]) -> List[RiderMatch]:
    persons = [d for d in scene if d.class_id == 0]
    bikes = [d for d in scene if d.class_id == 3]
    out: List[RiderMatch] = []

    for bike in bikes:
        bx1, by1, bx2, by2 = bike.box
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        candidates = []
        for p in persons:
            pcx, pcy = p.bottom_center
            # Rider must sit over/around the motorcycle, not merely be nearby.
            rider_zone = (bx1 - 0.65*bw, by1 - 2.3*bh, bx2 + 0.65*bw, by2 + 0.15*bh)
            if not _inside((pcx, pcy), rider_zone):
                continue
            bcx, bcy = bike.center
            score = abs(pcx-bcx)/bw + 0.35*abs(pcy-bcy)/bh
            candidates.append((score, p))
        if not candidates:
            continue
        person = min(candidates, key=lambda x: x[0])[1]

        px1, py1, px2, py2 = person.box
        ph = max(1, py2-py1)
        pw = max(1, px2-px1)
        head_zone = (px1-0.20*pw, py1-0.10*ph, px2+0.20*pw, py1+0.52*ph)
        hcands = [h for h in helmets if _inside(h.center, head_zone)]
        helmet = max(hcands, key=lambda h: h.confidence) if hcands else None
        out.append(RiderMatch(person, bike, helmet))
    return out


def associate_car_plates(scene: List[Detection], plates: List[Detection]) -> List[PlateMatch]:
    vehicles = [d for d in scene if d.class_id in (2, 5, 7)]
    out: List[PlateMatch] = []
    used = set()

    for vehicle in vehicles:
        vx1, vy1, vx2, vy2 = vehicle.box
        vw, vh = max(1, vx2-vx1), max(1, vy2-vy1)
        best = None
        for i, plate in enumerate(plates):
            if i in used:
                continue
            px1, py1, px2, py2 = plate.box
            pw, ph = max(1, px2-px1), max(1, py2-py1)
            pcx, pcy = plate.center
            if not _inside((pcx, pcy), vehicle.box, margin_x=0.05, margin_y=0.04):
                continue
            rel_y = (pcy-vy1)/vh
            width_ratio = pw/vw
            area_ratio = (pw*ph)/float(vw*vh)
            aspect = pw/float(ph)
            # Reject common false positives: windows, lamps, road markings.
            if not 0.06 <= width_ratio <= 0.75:
                continue
            if not 0.0008 <= area_ratio <= 0.16:
                continue
            if not 0.75 <= aspect <= 6.5:
                continue
            if rel_y < 0.38:
                continue
            score = abs(pcx-(vx1+vx2)/2)/vw + 0.4*abs(pcy-vy2)/vh - 0.15*plate.confidence
            if best is None or score < best[0]:
                best = (score, i, plate)
        if best is not None:
            _, i, plate = best
            used.add(i)
            plate.track_id = vehicle.track_id
            plate.extra = {**(plate.extra or {}), "vehicle": vehicle.label, "vehicle_track_id": vehicle.track_id}
            out.append(PlateMatch(vehicle, plate))
    return out
