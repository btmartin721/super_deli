"""Installed frozen-coordinate diagnostic with optional training/input imports blocked."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    root = parser.parse_args().output_dir.resolve()
    code = """
import sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','snpio'}:raise RuntimeError('Unexpected import: '+name)
sys.meta_path.insert(0,Block())
from pathlib import Path
from super_deli.benchmarking.coordinate_fixture import create_fixture
from super_deli.benchmarking.coordinate_config import CoordinateStudy
from super_deli.cli import main
from super_deli.io.config import write_json,read_json
root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
source=create_fixture(root/'source');out=root/'study';out.mkdir()
(out/'approved-plan.md').write_text('Invented installed engineering fixture only.')
write_json(CoordinateStudy(fixture=True).to_dict(),root/'config.json')
for action in ('bind','freeze','preflight','audit-preflight','gate','main','audit-main','report'):
 args=['diagnose-neural-coordinates',action,'--archive-root',str(source),'--output-dir',str(out),'--study-config',str(root/'config.json')]
 if action=='gate':args+=['--staging-bytes','0']
 assert main(args)==0
summary=read_json(out/'summary.json')
assert summary['neural_training']==0 and summary['new_pvalues']==0
assert 'records' not in summary['audits']['main']
print('Installed frozen-coordinate lifecycle passed without Torch or SNPio')
"""
    subprocess.run(
        [sys.executable, "-c", code, str(root)], check=True, capture_output=True, text=True
    )
    print("Installed frozen-coordinate lifecycle and optional-import guard passed")


if __name__ == "__main__":
    main()
