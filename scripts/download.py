"""Fetch only the official Optdigits train/test features, with provenance."""
from pathlib import Path
import urllib.request, zipfile, io, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip'
def main():
    req=urllib.request.Request(URL,headers={'User-Agent':'InkLabPortfolio/1.0'})
    with urllib.request.urlopen(req,timeout=90) as response: blob=response.read()
    out=ROOT/'data';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in ['optdigits.tra','optdigits.tes','optdigits.names']:
            matches=[p for p in archive.namelist() if p.rsplit('/',1)[-1]==name]
            if len(matches)!=1:raise ValueError('Missing expected file '+name)
            (out/name).write_bytes(archive.read(matches[0]))
    manifest={'source':'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','doi':'10.24432/C50P49','creators':['E. Alpaydin','C. Kaynak'],'year':1998,'license':'CC BY 4.0','archiveSha256':hashlib.sha256(blob).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='manifest.json'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Downloaded official training and different-writer test files.')
if __name__=='__main__':main()
