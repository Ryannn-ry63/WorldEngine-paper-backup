import json,pickle,sys
from pathlib import Path
from innovation3.scene_payload import single_scene_mapping
from worldengine.online.headless import HeadlessSimulator
row=json.loads(Path(sys.argv[1]).read_text())[0]
with open(row["scene_path"],"rb") as f:scene=single_scene_mapping(pickle.load(f),row["scene_id"])[row["scene_id"]]
reports=[]
for i in range(2):
 sim=HeadlessSimulator(row["scene_id"],scene,"R",18,row["scene_seed"],backend="cached")
 try:
  raw=sim.codec.describe(sim.snapshot());sim.engine._episode_start_time=0.0
  normalized=sim.codec.describe(sim.snapshot())
  reports.append(dict(raw=raw,normalized=normalized))
 finally:sim.close()
print(json.dumps(dict(raw_equal=reports[0]["raw"]["state_hash"]==reports[1]["raw"]["state_hash"],normalized_equal=reports[0]["normalized"]["state_hash"]==reports[1]["normalized"]["state_hash"],raw_component_differences=[k for k,v in reports[0]["raw"]["components"].items() if v!=reports[1]["raw"]["components"].get(k)],normalized_component_differences=[k for k,v in reports[0]["normalized"]["components"].items() if v!=reports[1]["normalized"]["components"].get(k)],reports=reports),indent=2))
