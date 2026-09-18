"""Same-state logged-future PDM labels. No policy/action intervention is allowed."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_attribution_common as a
from audit_selector_cfpi_deployment import action_arrays, action_error
from worldengine.manager.dense_reward_manager import DenseRewardManager
from worldengine.manager.diffusiondrive_dynamic_reward_manager import DiffusionDriveDynamicRewardManager
from worldengine.manager.diffusiondrive_preaction_oracle_manager import DiffusionDrivePreActionOracleManager


class AttributionLocalManager(DiffusionDriveDynamicRewardManager):
    _planner_inputs_preaction = DiffusionDrivePreActionOracleManager._planner_inputs_preaction
    _prepare_preaction_scores = DiffusionDrivePreActionOracleManager._prepare_preaction_scores
    step = DiffusionDrivePreActionOracleManager.step

    def __init__(self, observer):
        super().__init__()
        self.observer = observer
        contract = observer.collection
        if (contract['research_method'] != a.METHOD or contract['cohort'] != 'train'
                or contract['role'] != 'local_observation' or
                contract['local_definition'] != a.LOCAL_DEFINITION):
            raise RuntimeError('Local labels require the frozen training-only observer contract')
        targets = d.verified_read(contract['local_targets'])['targets']
        self.targets = {t['scene_id']: t for t in targets}
        if len(self.targets) != len(targets) or any(t['mode'] != contract['react_type'] for t in targets):
            raise RuntimeError('Duplicate scene or wrong local-label mode')
        if any('feedback_target' in r for r in observer.routing['routes']):
            raise RuntimeError('Local observer cannot perform interventions')
        if self.engine.global_config['reward_sampling_poses'] != 8:
            raise RuntimeError('Local proxy horizon must stay at four seconds')

    def reset(self):
        super().reset()
        self.observer.reset()

    def before_step(self):
        DenseRewardManager.before_step(self)
        self.observer.before_step()
        target = self.targets.get(str(self.current_scene['id']))
        if target is None:
            raise RuntimeError('Unregistered local-label scene')
        if self.current_step + 1 != target['decision_step']:
            return {}
        sidecar, path = self._load_sidecar()
        source = c.load_pickle(d.verify(target['context']))
        if (int(sidecar['planner_step']) != target['decision_step'] or
                int(sidecar['selected_index']) != target['visiting_index'] or
                f.fingerprint(sidecar) != target['source_fingerprint'] or
                any(not np.array_equal(np.asarray(sidecar[k]), np.asarray(source[k]))
                    for k in f.FINGERPRINT_FIELDS)):
            raise RuntimeError('Visited state/candidate/action does not match the frozen source')
        before = action_arrays(self.engine.external_actions)
        (scores, components), _ = self._prepare_preaction_scores(sidecar['candidate_trajectories_8'])
        if action_error(before, action_arrays(self.engine.external_actions)) != 0.:
            raise RuntimeError('Local scorer changed the executed action')
        outcomes = a.local_outcomes(scores, components, target['candidate_indices'])
        record = dict(status='PASS', method=a.METHOD, row_id=target['row_id'],
                      scene_id=target['scene_id'], mode=target['mode'],
                      decision_step=target['decision_step'], source_fingerprint=f.fingerprint(sidecar),
                      candidate_bank_sha=x.candidate_bank_hash(sidecar), selected_index=int(sidecar['selected_index']),
                      candidate_indices=target['candidate_indices'], pair_outcomes=outcomes,
                      scores=np.asarray(scores).tolist(), components=np.asarray(components).tolist(),
                      local_definition=a.LOCAL_DEFINITION, sidecar=d.artifact(path),
                      collection_sha256=self.observer.collection_sha, action_max_abs=0.,
                      simulator='TorchSimulator' if self.use_cuda else 'PDMSimulator',
                      inference_uses_labels=False)
        folder = Path(self.engine.global_config['data_output_dir'])/'attribution_local_records'
        c.atomic_json(folder/f"{target['row_id']}.json", record)
        return {}
