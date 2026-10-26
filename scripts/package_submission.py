"""Create a compact source/results archive without raw release data or notes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    destination=ROOT/'dist';destination.mkdir(exist_ok=True)
    files=[]
    excluded={'.venv','__pycache__','.pytest_cache','.git','dist','notes'}
    for path in ROOT.rglob('*'):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(part in excluded or part.endswith('.egg-info') for part in rel.parts):continue
        if path.name in {'.project.json','data-v1.tar.gz','SUBMISSION_SHA256.json'}:continue
        if rel.parts[0]=='challenge_data' and path.name!='README.md':continue
        files.append(path)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    archive=destination/'prince-maurya-atomity-forecasting.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'atomity-forecasting/'+path.relative_to(ROOT).as_posix())
        z.writestr('atomity-forecasting/SUBMISSION_SHA256.json',json.dumps(manifest,indent=2))
    (destination/'SUBMISSION_SHA256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'archive':str(archive),'files':len(files),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
