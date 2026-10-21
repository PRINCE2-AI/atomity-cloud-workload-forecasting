"""Download exact data-v1 and safely extract only its four required files."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request

URL='https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz'
FILES={'vm_cpu_readings.csv.gz','vm_metadata.csv.gz','DATA_CARD.md','SHA256SUMS'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='challenge_data');args=parser.parse_args()
    dest=Path(args.data);dest.mkdir(parents=True,exist_ok=True)
    archive=dest.parent/'data-v1.tar.gz'
    if not archive.exists():
        print('Downloading fixed data-v1 release (~116.5 MB)...')
        urllib.request.urlretrieve(URL,archive)
    with tarfile.open(archive,'r:gz') as tar:
        found=set()
        for member in tar.getmembers():
            name=Path(member.name).name
            if name in FILES and member.isfile():
                if name in found:raise ValueError('Duplicate archive member')
                found.add(name)
                source=tar.extractfile(member)
                with (dest/name).open('wb') as target:
                    import shutil
                    shutil.copyfileobj(source,target)
        if found!=FILES:raise ValueError('Incomplete release archive')
    for line in (dest/'SHA256SUMS').read_text().splitlines():
        if not line.strip():continue
        checksum,name=line.split(maxsplit=1);name=name.strip().lstrip('*')
        if name not in FILES:raise ValueError('Unexpected checksum path')
        actual=hashlib.sha256((dest/name).read_bytes()).hexdigest()
        if checksum!=actual:raise ValueError(f'Checksum mismatch: {name}')
        print(name+': OK')

if __name__=='__main__':main()
