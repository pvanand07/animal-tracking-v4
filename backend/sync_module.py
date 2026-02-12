"""Sync module: resolve AI detections to animals DB entries and create detection records."""
import logging
from database import (
    find_animal, insert_animal, create_detection,
    get_detection_by_tracking, get_ai_detection,
    _animal_creation_lock,
)
from ai_module import fetch_animal_info

log = logging.getLogger("sync_module")


def sync_detection(tracking_id: str) -> dict | None:
    """
    Given a tracking_id with a completed AI detection:
    1. Look up animal in animals table by common_name or scientific_name
    2. If not found → call LLM to fetch info → insert into animals
    3. Create a detection record linking tracking_id → animal_id
    """
    # Check if already synced
    existing = get_detection_by_tracking(tracking_id)
    if existing:
        log.debug(f"Detection already synced for {tracking_id}")
        return existing

    # Get AI detection result
    ai_det = get_ai_detection(tracking_id)
    if not ai_det:
        log.debug(f"No AI detection for {tracking_id}")
        return None

    # Normalize names so "Sheep"/" Sheep"/"Sheep " all resolve to the same row
    common_name = (ai_det.get("common_name") or "").strip() or None
    scientific_name = (ai_det.get("scientific_name") or "").strip() or None
    if not common_name and not scientific_name:
        log.debug("No common_name or scientific_name for sync")
        return None

    # Find in parallel (no lock)
    animal = find_animal(common_name=common_name, scientific_name=scientific_name)
    if not animal:
        # Only insert is under lock; re-find inside lock to avoid duplicate insert
        with _animal_creation_lock:
            animal = find_animal(common_name=common_name, scientific_name=scientific_name)
            if not animal:
                log.info(f"Animal '{common_name}' not in DB — fetching info via LLM")
                info = fetch_animal_info(common_name or "", scientific_name or "")
                if info:
                    animal_id = insert_animal(info)
                    log.info(f"Inserted animal '{common_name}' with id={animal_id}")
                    animal = {"id": animal_id, "animal": common_name or info.get("animal", "")}
                else:
                    log.warning(f"Could not fetch info for '{common_name}'")
                    return None

    display_name = common_name or animal.get("animal", "")
    
    # Create detection record
    det_id = create_detection(tracking_id, animal["id"])
    log.info(f"Created detection id={det_id}: {tracking_id} → animal_id={animal['id']}")

    return {
        "detection_id": det_id,
        "tracking_id": tracking_id,
        "animal_id": animal["id"],
        "animal": display_name,
    }
