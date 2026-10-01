"""Validate the exact scene/asset pair; never substitute a neighbouring asset."""
from pathlib import Path
from .paths import checked_path, sha256_file


def visual_asset(cfg, scenes):
    scene_id = cfg.get('visual_scene_id', sorted(scenes)[0])
    if scene_id not in scenes:
        raise ValueError('visual_scene_id is absent from the registered training source: ' + str(scene_id))
    asset_id = scenes[scene_id]['metadata'].get('digitaltwin_asset_id', scene_id)
    if not isinstance(asset_id, str) or not asset_id or Path(asset_id).name != asset_id or asset_id in ('.', '..'):
        raise ValueError('Invalid visual asset ID: ' + str(asset_id))
    # Match MTGSAssetManager's augmented-scene suffix convention.
    if asset_id[-4:].startswith('-'):
        asset_id = asset_id[:-4]
    # Formal scene manifests record an exact checkpoint path because public
    # assets do not live below the initialized selector's owned asset_root.
    # Keep the legacy root/scene lookup for engineering manifests.
    explicit = cfg.get('visual_asset_path')
    if explicit:
        asset = checked_path(explicit)
        expected_name = asset_id + '.ckpt'
        if (asset.name != expected_name or asset.parent.name != 'background' or
                asset.parent.parent.name != asset_id):
            raise ValueError(
                'visual_asset_path does not match the audited scene: ' +
                str(asset))
        root = asset.parent.parent.parent
        expected_sha = cfg.get('visual_asset_sha256')
        if expected_sha and sha256_file(asset) != expected_sha:
            raise ValueError('Visual checkpoint checksum mismatch: ' + str(asset))
    else:
        root = checked_path(cfg['asset_root'])
        expected = root / asset_id / 'background' / (asset_id + '.ckpt')
        try:
            asset = checked_path(expected)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                'Visual scene ' + scene_id + ' requires exact asset ' + str(expected) +
                '. asset_root must directly contain scene asset directories '
                '(for WE_processed training data: navtrain/assets). '
                'Select an explicitly audited visual_scene_id; no automatic substitution is performed.'
            ) from error
    if not asset.is_file() or asset.stat().st_size == 0:
        raise ValueError('Visual checkpoint is not a nonempty file: ' + str(asset))
    return scene_id, root, asset
