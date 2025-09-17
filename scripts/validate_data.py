import sys
sys.path.insert(0,'/lustre/fswork/projects/rech/jeb/uuz13rj/mount/beans')
from beans.utils import check_md5


with open('data/file_hashes') as f:
    for line in f:
        path, md5 = line.strip().split('\t')
        if path.startswith('data/cbi') :
            print(f'Validating {path} ...')
        #if not path.startswith('data//') or path.startswith('data/speech_commands/') :
            check_md5(path, md5)

print('Validation succeeded!')
