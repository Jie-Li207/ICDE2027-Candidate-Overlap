import argparse, json, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent

def training_plan():
    jobs = []
    for ds, seeds, device in [('FBDB15K/norm', range(3), 'gpu'), ('DBP15K/zh_en', range(3), 'cpu'), ('DBP15K/fr_en', [0], 'cpu'), ('FBYG15K/norm', [0], 'cpu'), ('DBP15K/fr_en', [1], 'gpu'), ('FBYG15K/norm', [1], 'gpu')]:
        for seed in seeds:
            for noimage in [False, True]:
                script = 'desalign_gpu.py' if device == 'gpu' else 'desalign_crosskg.py'
                folder = 'desalign_gpu' if device == 'gpu' else 'desalign_cpu'
                stem = ds.replace('/', '_') + f'_block{seed}' + ('_noimage' if noimage else '')
                jobs.append((folder + '/' + stem + '_epoch50.pt', [script, '--dataset', ds, '--block', str(seed)] + (['--no-image'] if noimage else [])))
    jobs.append(('cdmea_cpu/block0_epoch200.pt', ['cdmea_baseline.py', '--block', '0']))
    for ds, seed in [('FBDB15K/norm', 1), ('FBDB15K/norm', 2), ('DBP15K/zh_en', 0), ('DBP15K/fr_en', 0), ('FBYG15K/norm', 0)]:
        jobs.append(('cdmea_gpu/' + ds.replace('/', '_') + f'_block{seed}_epoch200.pt', ['cdmea_gpu.py', '--dataset', ds, '--block', str(seed)]))
    return jobs

def run(args):
    print('RUN', args, flush=True)
    subprocess.run([sys.executable, '-u'] + args, cwd=R, check=True)

def main():
    from settings import materialize
    materialize()
    a = argparse.ArgumentParser()
    a.add_argument('--stage', choices=['plan', 'analysis', 'verify', 'prepare', 'train', 'evaluate', 'systems', 'retrieval', 'retrieval-analysis'], default='plan')
    a.add_argument('--fresh-results', action='store_true', help='Preserve supplied outcomes in a timestamped archive before recomputing')
    args = a.parse_args()
    if args.stage == 'plan':
        print(json.dumps(training_plan(), indent=2))
        return
    if args.stage == 'analysis':
        for s in ['summarize_confirmatory.py', 'validate_paired_image.py', 'build_paper_evidence.py', 'validate_recorded_results.py', 'validate_candidate_recorded.py']:
            run([s])
        run(['reproduce.py', '--stage', 'retrieval-analysis'])
    elif args.stage == 'retrieval-analysis':
        for folder in ['retrieved_candidates', 'product_controls', 'product_text_boundary', 'twohop_queries', 'rank_fusion']:
            run(['validate_retrieved_candidates.py', folder])
        run(['mechanism.py'])
        run(['build_retrieval_evidence.py'])
        run(['query_diagnostics.py'])
        run(['build_diagnostic_figures.py'])
    elif args.stage == 'retrieval':
        paths = [R / p for p in ['retrieved_candidates', 'product_controls', 'overlap_mechanism', 'retrieval_cost', 'product_text_boundary', 'twohop_queries', 'rank_fusion']]
        if any((list(p.glob('*_results.json')) for p in paths)) and (not args.fresh_results):
            raise SystemExit('Recorded revision outcomes exist; use --fresh-results to archive and recompute, or --stage retrieval-analysis.')
        if args.fresh_results:
            archive = R / 'reproduction_history' / ('revision_' + str(int(time.time())))
            archive.mkdir(parents=True, exist_ok=False)
            for p in paths:
                if p.exists():
                    dest = archive / p.name
                    assert p.resolve().is_relative_to(R) and dest.resolve().is_relative_to(archive.resolve())
                    p.rename(dest)
        for s in ['download_products.py', 'retrieved_candidates.py', 'product_controls.py', 'validate_raw_facts.py', 'retrieval_cost.py']:
            run([s])
        run(['product_controls.py', '--exclude-shared-text'])
        run(['twohop_queries.py'])
        run(['rank_fusion.py'])
        run(['reproduce.py', '--stage', 'retrieval-analysis'])
    elif args.stage == 'verify':
        run(['bootstrap_research.py', '--verify-only'])
    elif args.stage == 'prepare':
        run(['bootstrap_research.py'])
    elif args.stage == 'train':
        for checkpoint, cmd in training_plan():
            if not (R / checkpoint).exists():
                run(cmd)
    elif args.stage == 'evaluate':
        if (R / 'confirmatory/results').exists() and (not args.fresh_results):
            raise SystemExit('Recorded outcomes exist. Use --fresh-results to archive them and actually recompute, or --stage analysis to replay recorded evidence.')
        if args.fresh_results:
            archive = R / 'reproduction_history' / str(int(time.time()))
            archive.mkdir(parents=True, exist_ok=False)
            paths = [R / p for p in ['confirmatory/results', 'confirmatory/direct', 'confirmatory/sql', 'candidate_continuation', 'candidate_controls', 'chance', 'tfp_fidelity', 'frozen', 'systems']]
            paths += list((R / 'confirmatory').glob('*_simple_complete.json')) + list((R / 'confirmatory').glob('*_cdmea_complete.json'))
            paths.append(R / 'paired_image')
            for p in paths:
                if not p.exists():
                    continue
                dest = archive / p.relative_to(R)
                assert p.resolve().is_relative_to(R) and dest.resolve().is_relative_to(archive.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dest)
        for s in ['export_frozen.py', 'confirmatory_data.py', 'confirmatory_eval.py', 'confirmatory_tfp.py', 'confirmatory_simple.py', 'confirmatory_direct.py', 'confirmatory_cdmea.py', 'confirmatory_sql.py', 'candidate_continuation.py', 'validate_candidate_sql.py', 'candidate_controls.py', 'tfp_fidelity.py']:
            run([s])
        run(['chance_control.py'])
        run(['export_candidate_workloads.py'])
        run(['validate_candidate_recorded.py'])
        run(['paired_image_boundary.py'])
        run(['validate_paired_image.py'])
    elif args.stage == 'systems':
        for ds in ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/fr_en']:
            m = json.loads((R / 'confirmatory' / (ds.replace('/', '_') + '_identities.json')).read_text())
            for n in [512, 1024, 2048, 4096, 8192]:
                if n <= len(m['test']):
                    run(['systems.py', '--dataset', ds, '--n', str(n)])
            run(['systems.py', '--dataset', ds, '--n', '512', '--expanded'])
if __name__ == '__main__':
    main()
