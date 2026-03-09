"""AI module: VLM identification + LLM animal info gathering via OpenRouter (built-in JSON mode)."""
import base64
import json
import logging
import numpy as np
import cv2
from openai import OpenAI
from config import config
from database import ANIMAL_COLUMNS

log = logging.getLogger("ai_module")

# OpenAI client for OpenRouter (uses /v1/chat/completions under the hood)
_openrouter_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _openrouter_client
    if config.openrouter_api_key == "":
        return None
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
        )
    return _openrouter_client

# ── VLM Identification (built-in JSON schema) ──────────────────

VLM_SYSTEM = "You are a wildlife expert. Respond ONLY with a valid JSON object. No other text, no markdown, no explanation. Use only the keys: common_name, scientific_name, description."

VLM_PROMPT = """Identify the animal in this image. If you can, output a JSON object with:
- common_name: common English name (or null)
- scientific_name: scientific/Latin name (or null)
- description: 1-2 sentence description
If it's not an animal (e.g. a human), or not identifiable, set common_name and scientific_name to null and description to "not identifiable"."""

VLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "common_name": {"type": ["string", "null"], "description": "Common English name of the animal"},
        "scientific_name": {"type": ["string", "null"], "description": "Scientific (Latin) name"},
        "description": {"type": "string", "description": "1-2 sentence description of what you see"},
    },
    "required": ["common_name", "scientific_name", "description"],
    "additionalProperties": False,
}

def identify_animal(frame_crop: np.ndarray) -> dict | None:
    """Send cropped frame to Qwen VLM for animal identification (JSON mode)."""
    client = _get_client()
    if client is None:
        log.warning("OPENROUTER_API_KEY not set — skipping VLM identification")
        return None

    success, buf = cv2.imencode(".jpg", frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return None
    b64_image = base64.b64encode(buf.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_image}"

    try:
        log.debug("VLM request: model=%s", config.vlm_model)
        response = client.chat.completions.create(
            model=config.vlm_model,
            max_tokens=300,
            messages=[
                {"role": "system", "content": VLM_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": VLM_PROMPT},
                    ],
                },
            ],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_identification",
                    "strict": True,
                    "schema": VLM_JSON_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            log.warning("VLM response empty content")
            return None
        result = json.loads(content)
        common_name = result.get("common_name")
        scientific_name = result.get("scientific_name")
        if common_name:
            log.info("VLM identified: %s (%s)", common_name, scientific_name or "?")
        return result
    except Exception as e:
        log.error("VLM identification error: %s", e)
        return None


# ── LLM Animal Info Gathering (built-in JSON schema) ─────────────

INFO_PROMPT_TEMPLATE = """You are a wildlife database assistant with web search access.
Look up detailed information about this animal: "{animal_name}" (scientific name: {scientific_name}).
Use strings for all values; use "unknown" if not found.
Include safety_info: brief guidance for human encounters (null if not an animal or if subject is a human).
Include is_dangerous: true if the animal can pose a threat to humans, false otherwise (false for humans).
"""

# JSON schema for animal info: all ANIMAL_COLUMNS as string properties
LLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "animal": {"type": "string", "description": "Common name of the animal"},
        "height_cm": {"type": "string", "description": "Average height in cm"},
        "weight_kg": {"type": "string", "description": "Average weight in kg"},
        "color": {"type": "string", "description": "Primary colors/patterns"},
        "lifespan_years": {"type": "string", "description": "Average lifespan in years"},
        "diet": {"type": "string", "description": "Carnivore/herbivore/omnivore + details"},
        "habitat": {"type": "string", "description": "Primary habitats"},
        "predators": {"type": "string", "description": "Main predators"},
        "average_speed_kmh": {"type": "string", "description": "Average speed in km/h"},
        "countries_found": {"type": "string", "description": "Countries where found"},
        "conservation_status": {"type": "string", "description": "IUCN status"},
        "family": {"type": "string", "description": "Taxonomic family"},
        "gestation_period_days": {"type": "string", "description": "Gestation in days"},
        "top_speed_kmh": {"type": "string", "description": "Top speed in km/h"},
        "social_structure": {"type": "string", "description": "Social behavior"},
        "offspring_per_birth": {"type": "string", "description": "Typical offspring count"},
        "scientific_name": {"type": "string", "description": "Scientific name"},
        "safety_info": {"type": ["string", "null"], "description": "Safety information for human encounters (null if not an animal)"},
        "is_dangerous": {"type": "boolean", "description": "Whether the animal is dangerous to humans (false if not an animal)"},
    },
    "required": list(ANIMAL_COLUMNS),
    "additionalProperties": False,
}


def fetch_animal_info(common_name: str, scientific_name: str = "unknown") -> dict | None:
    """Use LLM with JSON mode to gather animal information."""
    client = _get_client()
    if client is None:
        log.warning("OPENROUTER_API_KEY not set — skipping info gathering")
        return _fallback_animal_info(common_name, scientific_name)

    prompt = INFO_PROMPT_TEMPLATE.format(
        animal_name=common_name,
        scientific_name=scientific_name or "unknown",
    )

    try:
        response = client.chat.completions.create(
            model=config.llm_model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "animal_info",
                    "strict": True,
                    "schema": LLM_JSON_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            return _fallback_animal_info(common_name, scientific_name)
        result = json.loads(content)
        for col in ANIMAL_COLUMNS:
            if col not in result or result[col] is None:
                result[col] = False if col == "is_dangerous" else "unknown"
        result["animal"] = common_name
        if scientific_name and scientific_name != "unknown":
            result["scientific_name"] = scientific_name
        log.info("LLM gathered info for: %s", common_name)
        return result
    except json.JSONDecodeError as e:
        log.error("LLM info gathering JSON error: %s", e)
        return _fallback_animal_info(common_name, scientific_name)
    except Exception as e:
        log.error("LLM info gathering error: %s", e)
        return _fallback_animal_info(common_name, scientific_name)


def _fallback_animal_info(common_name: str, scientific_name: str = "unknown") -> dict:
    """Minimal record when API is unavailable."""
    info = {col: ("unknown" if col != "is_dangerous" else False) for col in ANIMAL_COLUMNS}
    info["animal"] = common_name
    info["scientific_name"] = scientific_name or "unknown"
    return info
