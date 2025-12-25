import subprocess
import requests

token = subprocess.run([r'C:\Program Files\GitHub CLI\gh.exe', 'auth', 'token'], capture_output=True, text=True).stdout.strip()

response = requests.get(
    'https://models.inference.ai.azure.com/models',
    headers={'Authorization': f'Bearer {token}'}
)

if response.status_code == 200:
    data = response.json()
    print('Available models:')
    if isinstance(data, list):
        for m in data:
            if isinstance(m, dict):
                model_id = m.get('id') or m.get('name') or str(m)
                print(f"  - {model_id}")
            else:
                print(f"  - {m}")
    elif isinstance(data, dict):
        models = data.get('data') or data.get('models') or [data]
        for m in models:
            if isinstance(m, dict):
                model_id = m.get('id') or m.get('name') or str(m)
                print(f"  - {model_id}")
else:
    print(f'Error {response.status_code}: {response.text[:500]}')
