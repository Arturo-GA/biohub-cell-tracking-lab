"""Create the requested private repo using the existing credential helper.

Credentials remain in memory; only the public repository receipt is saved.
"""
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def main():
    result = subprocess.run(['git', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True,
        capture_output=True, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    token = values.get('password')
    if not token:
        raise SystemExit('No existing GitHub credential available')

    def api(path, payload=None):
        request = urllib.request.Request('https://api.github.com' + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json', 'User-Agent': 'biohub-cell-tracking-lab'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    login = api('/user')['login']
    name = 'biohub-cell-tracking-lab'
    try:
        repo = api(f'/repos/{login}/{name}')
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        repo = api('/user/repos', {'name': name, 'private': True, 'auto_init': False,
            'description': 'Biohub Kaggle research, pinned public baselines, official evaluation and tracking experiments'})
    if not repo['private']:
        raise SystemExit('Existing repository is public; inspect before publishing competition work')
    receipt = {key: repo[key] for key in ('full_name', 'html_url', 'clone_url', 'private')}
    Path('results').mkdir(exist_ok=True)
    Path('results/github.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f'GitHub HTTP status {error.code}') from None
