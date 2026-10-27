import argparse, json, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent

def training_plan():
    jobs = []
    for ds, seeds, device in [('FBDB15K/norm', range(3), 'gpu'), ('DBP15K/zh_en', range(3), 'cpu'), ('DBP15K/fr_en', [0], 'cpu'), ('FBYG15K/norm', [0], 'cpu'), ('DBP15K/fr_en', [1], 'gpu'), ('FBYG15K/norm', [1], 'gpu')]:
        for seed in seeds:
            for noimage in [False, True]:
                script = 'desalign_gpu_v21.py' if device == 'gpu' else 'desalign_crosskg_v11.py'
                folder = 'desalign_v21_gpu' if device == 'gpu' else 'desalign_v11'
                stem = ds.replace('/', '_') + f'_block{seed}' + ('_noimage' if noimage else '')
                jobs.append((folder + '/' + stem + '_epoch50.pt', [script, '--dataset', ds, '--block', str(seed)] + (['--no-image'] if noimage else [])))
    jobs.append(('cdmea_v14/block0_epoch200.pt', ['cdmea_baseline_v14.py', '--block', '0']))
    for ds, seed in [('FBDB15K/norm', 1), ('FBDB15K/norm', 2), ('DBP15K/zh_en', 0), ('DBP15K/fr_en', 0), ('FBYG15K/norm', 0)]:
        jobs.append(('cdmea_v23_gpu/' + ds.replace('/', '_') + f'_block{seed}_epoch200.pt', ['cdmea_gpu_v23.py', '--dataset', ds, '--block', str(seed)]))
    return jobs

def run(args):
    print('RUN', args, flush=True)
    subprocess.run([sys.executable, '-u'] + args, cwd=R, check=True)

def main():
    from settings import materialize
    materialize()
    a = argparse.ArgumentParser()
    a.add_argument('--stage', choices=['plan', 'analysis', 'verify', 'prepare', 'train', 'evaluate', 'systems', 'revision', 'revision-analysis'], default='plan')
    a.add_argument('--fresh-results', action='store_true', help='Preserve supplied outcomes in a timestamped archive before recomputing')
    args = a.parse_args()
    if args.stage == 'plan':
        print(json.dumps(training_plan(), indent=2))
        return
    if args.stage == 'analysis':
        for s in ['summarize_confirmatory_v27.py', 'validate_paired_image_v33.py', 'build_paper_evidence.py', 'validate_recorded_results.py', 'validate_candidate_recorded.py']:
            run([s])
        run(['reproduce.py', '--stage', 'revision-analysis'])
    elif args.stage == 'revision-analysis':
        for folder in ['revision_v34', 'revision_v35', 'revision_v38', 'revision_v39', 'revision_v41']:
            run(['validate_revision_v34.py', folder])
        run(['mechanism_v36.py'])
        run(['build_revision_evidence.py'])
        run(['review_diagnostics_v40.py'])
        run(['build_review_evidence_v42.py'])
    elif args.stage == 'revision':
        paths = [R / p for p in ['revision_v34', 'revision_v35', 'revision_v36', 'revision_cost_v37', 'revision_v38', 'revision_v39', 'revision_v41']]
        if any((list(p.glob('*_results.json')) for p in paths)) and (not args.fresh_results):
            raise SystemExit('Recorded revision outcomes exist; use --fresh-results to archive and recompute, or --stage revision-analysis.')
        if args.fresh_results:
            archive = R / 'reproduction_history' / ('revision_' + str(int(time.time())))
            archive.mkdir(parents=True, exist_ok=False)
            for p in paths:
                if p.exists():
                    dest = archive / p.name
                    assert p.resolve().is_relative_to(R) and dest.resolve().is_relative_to(archive.resolve())
                    p.rename(dest)
        for s in ['download_products_v35.py', 'revision_v34.py', 'revision_products_v35.py', 'validate_revision_raw_facts.py', 'revision_cost_v37.py']:
            run([s])
        run(['revision_products_v35.py', '--exclude-shared-text'])
        run(['revision_twohop_v39.py'])
        run(['revision_fusion_v41.py'])
        run(['reproduce.py', '--stage', 'revision-analysis'])
    elif args.stage == 'verify':
        run(['bootstrap_research.py', '--verify-only'])
    elif args.stage == 'prepare':
        run(['bootstrap_research.py'])
    elif args.stage == 'train':
        for checkpoint, cmd in training_plan():
            if not (R / checkpoint).exists():
                run(cmd)
    elif args.stage == 'evaluate':
        if (R / 'confirmatory_v27/results').exists() and (not args.fresh_results):
            raise SystemExit('Recorded outcomes exist. Use --fresh-results to archive them and actually recompute, or --stage analysis to replay recorded evidence.')
        if args.fresh_results:
            archive = R / 'reproduction_history' / str(int(time.time()))
            archive.mkdir(parents=True, exist_ok=False)
            paths = [R / p for p in ['confirmatory_v27/results', 'confirmatory_v27/direct', 'confirmatory_v27/sql', 'candidate_v29', 'candidate_v31', 'chance_v32', 'tfp_fidelity_v30', 'frozen_v27', 'systems_v28']]
            paths += list((R / 'confirmatory_v27').glob('*_simple_complete.json')) + list((R / 'confirmatory_v27').glob('*_cdmea_complete.json'))
            paths.append(R / 'paired_image_v33')
            for p in paths:
                if not p.exists():
                    continue
                dest = archive / p.relative_to(R)
                assert p.resolve().is_relative_to(R) and dest.resolve().is_relative_to(archive.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dest)
        for s in ['export_frozen_v27.py', 'confirmatory_data_v27.py', 'confirmatory_eval_v27.py', 'confirmatory_tfp_v27.py', 'confirmatory_simple_v27.py', 'confirmatory_direct_v27.py', 'confirmatory_cdmea_v27.py', 'confirmatory_sql_v27.py', 'candidate_continuation_v29.py', 'validate_candidate_sql_v29.py', 'candidate_controls_v31.py', 'tfp_fidelity_v30.py']:
            run([s])
        run(['chance_control_v32.py'])
        run(['export_candidate_workloads.py'])
        run(['validate_candidate_recorded.py'])
        run(['paired_image_boundary_v33.py'])
        run(['validate_paired_image_v33.py'])
    elif args.stage == 'systems':
        for ds in ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']:
            m = json.loads((R / 'confirmatory_v27' / (ds.replace('/', '_') + '_identities.json')).read_text())
            for n in [512, 1024, 2048, 4096, 8192]:
                if n <= len(m['test']):
                    run(['systems_v28.py', '--dataset', ds, '--n', str(n)])
            run(['systems_v28.py', '--dataset', ds, '--n', '512', '--expanded'])
if __name__ == '__main__':
    main()
