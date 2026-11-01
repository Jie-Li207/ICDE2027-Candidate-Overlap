import argparse
import subprocess
import sys
from pathlib import Path
R = Path(__file__).resolve().parent

def run(script, *args):
    subprocess.run([sys.executable, '-u', script, *args], cwd=R, check=True)

def main():
    from settings import materialize
    materialize()
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['retail', 'graph', 'validate', 'analysis', 'all'], default='all')
    stage = parser.parse_args().stage
    if stage in ['retail', 'all']:
        for script in ['fetch_retail.py', 'encode_retail.py', 'retail_scope.py']:
            run(script)
    if stage in ['graph', 'all']:
        run('query_scope.py')
        run('cdmea_retrieval.py', '--stage', 'export')
        run('cdmea_retrieval.py', '--stage', 'evaluate')
    if stage in ['validate', 'all']:
        run('validate_scope.py')
        run('audit_permutations.py')
    if stage in ['analysis', 'all']:
        run('analyze_scope.py')
        run('plot_scope.py')
if __name__ == '__main__':
    main()
