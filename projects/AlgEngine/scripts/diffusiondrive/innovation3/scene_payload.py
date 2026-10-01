"""Small, dependency-free normalization for audited individual scene files."""


def single_scene_mapping(loaded, scene_id):
    """Normalize ``{scene_id: scene}`` and direct scene dictionaries."""
    if not isinstance(loaded, dict) or not isinstance(scene_id, str) or not scene_id:
        raise ValueError('Single-scene pickle must be a dictionary and requires visual_scene_id')
    candidate = loaded.get(scene_id)
    if isinstance(candidate, dict):
        scene = candidate
    elif loaded.get('id') == scene_id or loaded.get('name') == scene_id:
        scene = loaded
    else:
        raise ValueError('Single-scene pickle does not contain requested scene: ' + scene_id)
    return {scene_id: scene}
