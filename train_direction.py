import argparse
import contextlib
import copy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

import preprocessing as features
from preprocessing import prepare_stage,load_independent_inputs,load_background_associations
from utils_direction import (ROOT,DATASETS,DEFAULT_GRID,METRICS,read_json,write_json,
    sha256,write_pairs,parse_grid,verify_sources,set_random_seed,evaluate_multiclass_from_logits,
    evaluate_multiclass_from_probs)

PAPER_FIXED = {
    'hidden': 256, 'hyper_pca_dim': 128, 'weight_decay': 1e-7,
    'pair_num_experts': 4, 'pair_topk_experts': 2,
    'branch_fusion_tau': 3.0, 'branch_residual_scale': .1,
    'w_cd': 1., 'w_up': 1., 'w_down': 1., 'w_cm': 1., 'w_md': 1., 'w_bridge': .5,
    'known_loss_lambda': 1., 'direction_loss_lambda': 1.5, 'tri_loss_lambda': .3,
    'down_direction_weight': 2., 'hyper_reg_lambda': 1.,
    'use_direction_supcon': 1, 'supcon_lambda': .2, 'supcon_temperature': .2,
    'supcon_mode': 'direction_only', 'supcon_class_balance': 1, 'supcon_warmup_ratio': .2,
    'loss_type': 'ce', 'class_weight_mode': 'balanced', 'use_amp': False,
    'graph_build_mode': 'full_direction_prior', 'direction_branch_mode': 'three_branch',
    'use_relation_scorer': 0, 'rel_loss_lambda': 0.,
}

def build_training_defaults(dataset):

    return argparse.Namespace(**PAPER_FIXED, pair_router_lambda=0. if dataset=='Data1' else .01,
        paper_dataset=dataset,epochs=500,lr=.0001,drop_rate=.1,
        supcon_min_samples_per_class=2,supcon_max_samples=4096)


def classification_weights(labels,num_classes,extra_pos_weight=1.):

    counts=np.bincount(labels.detach().cpu().numpy(),minlength=num_classes)
    safe=np.maximum(counts,1)
    weights=(safe.sum()/(num_classes*safe)).astype(np.float32)
    if num_classes==2:
        weights[1]*=extra_pos_weight
    return torch.as_tensor(weights,device=labels.device)


def hierarchical_loss(head,labels):

    directional=labels>0
    known_target=directional.long()
    known=F.cross_entropy(head['known_logits'],known_target,
        weight=classification_weights(known_target,2))
    direction_target=labels[directional]-1
    direction=F.cross_entropy(head['direction_logits'][directional],direction_target,
        weight=classification_weights(direction_target,2,2.)) if directional.any() else known.new_zeros(())
    tri=F.cross_entropy(head['final_logits'],labels,weight=classification_weights(labels,3))
    return known+1.5*direction+.3*tri,{'known_loss':known,'direction_loss':direction,'tri_loss':tri}


def supervised_contrastive_loss(features,labels,temperature=.2,max_samples=4096):

    keep=labels>0
    features,labels=features[keep],labels[keep]
    unique,counts=torch.unique(labels,return_counts=True)
    valid=unique[counts>=2]
    keep=(labels[:,None]==valid[None,:]).any(1)
    features,labels=features[keep],labels[keep]
    if len(labels)<2:
        return features.new_zeros(())
    if len(labels)>max_samples:
        chosen=torch.randperm(len(labels),device=labels.device)[:max_samples]
        features,labels=features[chosen],labels[chosen]
    features=F.normalize(features.float(),dim=1)
    logits=features @ features.T / temperature
    logits=logits-logits.max(1,keepdim=True).values.detach()
    off_diagonal=torch.ones_like(logits)
    off_diagonal.fill_diagonal_(0.)
    positive=(labels[:,None]==labels[None,:]).float()*off_diagonal
    positive_count=positive.sum(1)
    valid=positive_count>0
    if not valid.any():
        return logits.new_zeros(())
    log_probability=logits-torch.log((logits.exp()*off_diagonal).sum(1,keepdim=True).clamp_min(1e-8))
    anchor_loss=-(positive*log_probability).sum(1)/positive_count.clamp_min(1.)
    return torch.stack([anchor_loss[valid & (labels==c)].mean() for c in torch.unique(labels[valid])]).mean()


def invoke_training(args,fold_id,device,prepared,train,evaluate,folder,role,training_seed):

    from model_direction import DDHfgCDA
    from primary_metrics import save_primary_metrics
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    features.assert_disjoint(train,evaluate,(prepared[0].shape[0],prepared[1].shape[0]))
    set_random_seed(training_seed,deterministic=True)
    model=DDHfgCDA(args.hyper_pca_dim,prepared[2]['all'].shape[1],prepared[3]['all'].shape[1],
        drop_rate=args.drop_rate,embed_dim=args.hidden,
        num_experts=args.pair_num_experts,topk_experts=args.pair_topk_experts).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    train_index=torch.as_tensor(train[:,:2],dtype=torch.long,device=device)
    train_labels=torch.as_tensor(train[:,2],dtype=torch.long,device=device)
    if not (torch.bincount(train_labels,minlength=3)>0).all():
        raise ValueError('Each training stage requires all three classes')
    write_json(folder/'training_config.json',{'args':vars(args),'role':role,'seed':training_seed,
        'training_pairs_sha256':features.array_hash(train),'evaluation_pairs_sha256':features.array_hash(evaluate),
        'test_guided_selection':False})
    records=[]
    start=time.time()
    for epoch in range(1,args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits,head=model(*prepared[:4],train_index,return_aux=True)
        classification,terms=hierarchical_loss(head,train_labels)
        supcon=supervised_contrastive_loss(head['contrast_z'],train_labels,
            temperature=args.supcon_temperature,max_samples=args.supcon_max_samples)
        hyper=model.get_hyper_regularization_loss()
        router=model.get_pair_router_aux_loss()
        warmup=min(1.,epoch/max(1,int(.2*args.epochs)))
        loss=classification+warmup*args.supcon_lambda*supcon+warmup*args.hyper_reg_lambda*hyper+args.pair_router_lambda*router
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError(f'Nonfinite training state at epoch {epoch}')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.)
        optimizer.step()
        records.append({'epoch':epoch,'loss':float(loss.detach()),
            **{key:float(value.detach()) for key,value in terms.items()},
            'supcon_loss':float(supcon.detach()),'hyper_loss':float(hyper.detach()),
            'router_loss':float(router.detach()),'lambda_supcon':warmup*args.supcon_lambda,
            'lambda_hyper':warmup*args.hyper_reg_lambda,'lr':args.lr})
        if epoch==1 or epoch%100==0 or epoch==args.epochs:
            print(f'{args.paper_dataset} fold={fold_id+1} {role} epoch={epoch}/{args.epochs} loss={records[-1]["loss"]:.6f}',flush=True)
    pd.DataFrame(records).to_csv(folder/'training_log.csv',index=False)
    torch.save({'epoch':args.epochs,'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),'args':vars(args)},folder/'final_model.pth')
    model.eval()
    with torch.no_grad():
        evaluation_index=torch.as_tensor(evaluate[:,:2],dtype=torch.long,device=device)
        logits=model(*prepared[:4],evaluation_index)
    metrics,_,probabilities,predicted=evaluate_multiclass_from_logits(evaluate[:,2],logits)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1.,atol=1e-5):
        raise FloatingPointError('Invalid evaluation probabilities')
    prediction_frame=pd.DataFrame({'circ_index':evaluate[:,0],'disease_index':evaluate[:,1],
        'y_true':evaluate[:,2],'y_pred':predicted,'prob_unknown':probabilities[:,0],
        'prob_up':probabilities[:,1],'prob_down':probabilities[:,2]})
    prediction_frame.to_csv(folder/f'{role}_predictions.csv',index=False)
    write_json(folder/'metrics.json',metrics)
    save_primary_metrics(metrics,folder,f'{args.paper_dataset} / {role}')
    write_json(folder/'training_audit.json',{'role':role,'epochs_completed':args.epochs,
        'final_epoch_model_used':True,'model_reinitialized':True,'test_guided_selection':False,
        'training_seed':training_seed,'train_n':len(train),'evaluation_n':len(evaluate),
        'model_selection_uses_this_evaluation':role=='validation',
        'final_model_sha256':sha256(folder/'final_model.pth'),'elapsed_sec':time.time()-start})
    del model,optimizer,logits
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return metrics


def source_inventory(dataset,input_files,options):

    pairs=Path(options.data_root)/f'step5_training_samples_{DATASETS[dataset]["suffix"]}'
    files=list(ROOT.glob('*.py'))+list(input_files)
    files += [pairs/f'fold_{fold}'/f'{role}_pairs.npy' for fold in range(5) for role in ('train','test')]
    return {str(path.resolve()):sha256(path) for path in files}


def select_parameters(defaults, device, prepared, train, valid, candidates, options, folder):

    records = []
    for candidate_id, parameters in enumerate(candidates):
        args = copy.deepcopy(defaults)
        for key, value in parameters.items():
            setattr(args, key, value)
        metrics = invoke_training(args, options.fold_id, device, prepared, train, valid,
            folder / f'candidate_{candidate_id:03d}', 'validation', options.training_seed)
        record = {'candidate_id': candidate_id, **parameters,
                  'validation_macro_aupr': metrics['macro_aupr_ovr'],
                  'validation_macro_auc': metrics['macro_auc_ovr'],
                  'validation_macro_f1': metrics['macro_f1']}
        if not np.isfinite(record[options.selection_metric]):
            raise ValueError('Nonfinite validation selection score')
        records.append(record)
        pd.DataFrame(records).to_csv(folder / 'validation_search.csv', index=False)
        print(f'{options.dataset} outer={options.fold_id+1} candidate={candidate_id+1}/{len(candidates)} '
              f'{parameters} validation={record[options.selection_metric]:.6f}', flush=True)
    best = min(records, key=lambda row: (-row[options.selection_metric], row['epochs'], row['candidate_id']))
    selected = candidates[best['candidate_id']]
    write_json(folder.parent / 'selected_parameters.json', {
        'selected_parameters': selected, 'selected_candidate_id': best['candidate_id'],
        'selection_metric': options.selection_metric, 'selected_validation_value': best[options.selection_metric],
        'tie_break': 'fewer epochs, then declared candidate order',
        'candidate_count': len(candidates), 'test_labels_used_for_selection': False,
        'inner_training_pairs_sha256': features.array_hash(train),
        'inner_validation_pairs_sha256': features.array_hash(valid)})
    return selected

def run_outer(defaults, device, dss, cm, md, background, candidates, options, output):

    shape = DATASETS[options.dataset]['shape']
    source = Path(options.data_root) / f'step5_training_samples_{DATASETS[options.dataset]["suffix"]}' / f'fold_{options.fold_id}'
    folder = output / f'outer_fold_{options.fold_id+1}'
    folder.mkdir()
    outer_train = features.validate_pairs(np.load(source / 'train_pairs.npy', allow_pickle=False), shape)
    indices = np.arange(len(outer_train))
    tr, va = train_test_split(indices, test_size=options.validation_ratio,
                             stratify=outer_train[:, 2], random_state=options.split_seed)
    train, valid = outer_train[tr], outer_train[va]
    features.assert_disjoint(train, valid, shape)
    if not all((np.bincount(pairs[:, 2], minlength=3) > 0).all() for pairs in (train, valid)):
        raise ValueError('Each internal split must include all three classes')
    inner = folder / 'inner_selection'
    write_pairs(folder, 'outer_train', outer_train)
    write_pairs(inner, 'inner_train', train)
    write_pairs(inner, 'inner_validation', valid)
    inner_args = copy.deepcopy(defaults)
    with (inner / 'preprocessing.log').open('w') as log, contextlib.redirect_stdout(log):
        prepared = prepare_stage(inner_args, device, train, shape, cm, md,
                                          dss, background, options.preprocessing_seed, inner / 'preprocessing')
    features.check_heldout_edges(inner / 'preprocessing', valid)
    selection_hash, selected = None, None
    if not options.prepare_only:
        selected = select_parameters(inner_args, device, prepared, train, valid,
                                     candidates, options, inner)
        selection_hash = sha256(folder / 'selected_parameters.json')
    del prepared
    final_args = copy.deepcopy(defaults)
    for key, value in (selected or {}).items():
        setattr(final_args, key, value)
    with (folder / 'refit_preprocessing.log').open('w') as log, contextlib.redirect_stdout(log):
        final_prepared = prepare_stage(final_args, device, outer_train, shape, cm, md,
                                               dss, background, options.preprocessing_seed, folder / 'refit_preprocessing')

    test = features.validate_pairs(np.load(source / 'test_pairs.npy', allow_pickle=False), shape)
    for training in (outer_train, train, valid):
        features.assert_disjoint(training, test, shape)
    for stage in (inner / 'preprocessing', folder / 'refit_preprocessing'):
        features.check_heldout_edges(stage, test)
    write_pairs(folder, 'outer_test', test)
    audit = {'dataset': options.dataset, 'outer_fold': options.fold_id+1,
        'outer_training_n': len(outer_train), 'inner_training_n': len(train),
        'inner_validation_n': len(valid), 'outer_test_n': len(test),
        'all_class_pair_overlap_checks_passed': True,
        'inner_validation_edges_in_inner_relations': 0,
        'outer_test_edges_in_inner_and_refit_relations': 0,
        'cfs_and_gip_recomputed_for_inner_training_and_outer_refit': True,
        'refit_uses_all_outer_training_pairs': True,
        'permitted_background_edges': int(background.sum()),
        'adaptive_fusion_weights_recomputed_per_stage': True,
        'split_seed': options.split_seed, 'preprocessing_seed': options.preprocessing_seed,
        'training_seed': options.training_seed, 'prepare_only': options.prepare_only,
        'protocol': 'pairwise transductive', 'selected_parameters': selected,
        'outer_test_evaluations': 0}
    if options.prepare_only:
        write_json(folder / 'outer_audit.json', audit)
        print(f'PREPROCESSING CHECK PASSED {options.dataset} outer={options.fold_id+1}', flush=True)
        return audit
    metrics = invoke_training(final_args, options.fold_id, device, final_prepared,
        outer_train, test, folder / 'final_refit', 'test', options.training_seed)
    if sha256(folder / 'selected_parameters.json') != selection_hash:
        raise AssertionError('Selected configuration changed during test evaluation')
    audit.update(outer_test_evaluations=1, final_model_reinitialized=True,
                 test_labels_used_for_selection=False,
                 selected_parameters_sha256_before_test=selection_hash,
                 selected_parameters_sha256_after_test=sha256(folder / 'selected_parameters.json'))
    write_json(folder / 'outer_audit.json', audit)
    write_json(folder / 'outer_metrics.json', metrics)
    names = ['selected_parameters.json', 'outer_audit.json', 'outer_metrics.json',
             'final_refit/test_predictions.csv', 'final_refit/training_log.csv', 'final_refit/final_model.pth',
             'inner_selection/preprocessing/feature_audit.json', 'refit_preprocessing/feature_audit.json']
    write_json(folder / 'completed.json', {'sha256': {name: sha256(folder / name) for name in names}})
    print(f'COMPLETED {options.dataset} outer={options.fold_id+1}', flush=True)
    return {'dataset': options.dataset, 'outer_fold': options.fold_id+1, **metrics}

def summarize_dataset(output, dataset, records, folds):

    from utils_direction import evaluate_multiclass_from_probs
    frame = pd.DataFrame(records)
    summary = {'dataset': dataset, 'outer_folds': len(folds)}
    for key, label in METRICS.items():
        summary[label + '_mean'] = float(frame[key].mean())
        summary[label + '_std'] = float(frame[key].std(ddof=1)) if len(folds) > 1 else None
    pd.DataFrame([summary]).to_csv(output / 'metrics_table.csv', index=False)
    predictions = pd.concat([pd.read_csv(output / f'outer_fold_{fold}/final_refit/test_predictions.csv')
        .assign(outer_fold=fold) for fold in folds], ignore_index=True)
    if predictions.duplicated(['circ_index', 'disease_index']).any():
        raise AssertionError('Duplicate pairs across OOF predictions')
    predictions.to_csv(output / 'oof_predictions.csv', index=False)
    from primary_metrics import save_primary_fold_metrics, save_primary_metrics
    save_primary_fold_metrics([dict(record, fold=record['outer_fold']) for record in records], output, dataset)
    metrics, _ = evaluate_multiclass_from_probs(predictions.y_true,
        predictions[['prob_unknown', 'prob_up', 'prob_down']].to_numpy(), predictions.y_pred)
    write_json(output / 'pooled_oof_metrics.json', metrics)
    save_primary_metrics(metrics, output / 'pooled_oof', f'{dataset} / pooled OOF')
    return summary

def validate_options(options):

    folds = [int(value) for value in options.outer_folds.split(',')]
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(1, 6)):
        raise ValueError('outer-folds must contain unique 1-based fold numbers')
    if options.validation_ratio != .2 or options.selection_metric != 'validation_macro_aupr':
        raise ValueError('The manuscript fixes 20% validation and Macro-AUPR selection')
    grid = read_json(options.grid_json) if options.grid_json else copy.deepcopy(DEFAULT_GRID)
    candidates = parse_grid(grid)
    return folds, grid, candidates

def run(options):

    folds, grid, candidates = validate_options(options)
    dss, cm, md, input_files, provenance = load_independent_inputs(options.dataset, options.data_root, options.source_root)
    background, background_files, background_audit = load_background_associations(options.dataset, options.source_root, options.data_root)
    input_files.extend(background_files)
    if options.check_inputs:
        return {'dataset': options.dataset, 'inputs_valid': True,
                'candidate_count': len(candidates), 'outer_folds': folds,
                'background_audit': background_audit, 'DSS_shape': list(dss.shape), 'auxiliary_shapes': [list(cm.shape), list(md.shape)]}
    import torch
    if options.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    output = Path(options.output).resolve() / options.dataset
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    defaults = build_training_defaults(options.dataset)
    device = torch.device('cuda:0' if options.device == 'cuda' else 'cpu')
    inventory = source_inventory(options.dataset, input_files, options)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', {
        'dataset': options.dataset, 'outer_folds_1based': folds, 'grid': grid,
        'candidate_count_per_outer_fold': len(candidates), 'validation_ratio': options.validation_ratio,
        'selection_metric': options.selection_metric, 'prepare_only': options.prepare_only,
        'protocol': 'outer five-fold; inner single stratified validation; pairwise transductive',
        'fusion': {'circRNA': 'consistency-adaptive, Eq. (3)-(4)', 'disease': 'degree-aware, Eq. (2)'},
        'background_audit': background_audit, 'paper_fixed_parameters': PAPER_FIXED,
        'model_equations': 'differentiable propagation; current-stage aggregation each step; Eq. (15)-(39)',
        'regularization_detail': 'structure penalties 0.001/0.005/0.001; prototype consistency 0.01; see publication notes',
        'cfs_empty_set_rule': features.EMPTY_CFS_RULE, 'independent_input_provenance': provenance,
        'target_relations': 'permitted background plus this stage training pairs with labels 1 and 2',
        'base_args': vars(defaults), 'device': str(device),
        'split_seed_base': options.split_seed, 'preprocessing_seed_base': options.preprocessing_seed,
        'training_seed_base': options.training_seed, 'inputs_restricted_to_stage_training_and_independent_sources': True,
        'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'sklearn_version': __import__('sklearn').__version__,
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None})
    write_json(output / 'protected_source_hashes.json', inventory)
    records = []
    for fold in folds:
        stage_options = copy.copy(options)
        stage_options.fold_id = fold - 1
        stage_options.split_seed = options.split_seed + fold - 1
        stage_options.preprocessing_seed = options.preprocessing_seed + fold - 1
        stage_options.training_seed = options.training_seed + fold - 1
        records.append(run_outer(defaults, device, dss, cm, md, background, candidates, stage_options, output))
        verify_sources(inventory)
        if not options.prepare_only:
            pd.DataFrame(records).to_csv(output / 'outer_fold_metrics.csv', index=False)
    if options.prepare_only:
        write_json(output / 'preprocessing_validation.json', {
            'passed': True, 'outer_folds': records, 'model_training_runs': 0,
            'protected_sources_unchanged': True})
        return {'dataset': options.dataset, 'preprocessing_passed': True, 'model_training_runs': 0}
    summary = summarize_dataset(output, options.dataset, records, folds)
    write_json(output / 'run_completed.json', {
        'completed_outer_folds': folds, 'candidate_training_runs': len(folds)*len(candidates),
        'final_training_runs': len(folds), 'test_labels_used_for_selection': False,
        'cfs_and_gip_stage_local': True, 'protected_sources_unchanged': True,
        'completed_unix': time.time()})
    return summary

def build_parser(dataset_choices=('Data1', 'Data2', 'both'), default_dataset='both'):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=dataset_choices, default=default_dataset,
                        required=default_dataset is None)
    parser.add_argument('--source-root', default=str(ROOT), help='Raw association and auxiliary relation directory')
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--output', default=str(ROOT / 'results_cfs_paper'))
    parser.add_argument('--outer-folds', default='1,2,3,4,5')
    parser.add_argument('--grid-json', help='Predeclared hyperparameter/epoch grid')
    parser.add_argument('--validation-ratio', type=float, default=.2)
    parser.add_argument('--selection-metric', choices=['validation_macro_aupr', 'validation_macro_auc',
                                                    'validation_macro_f1'], default='validation_macro_aupr')
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--preprocessing-seed', type=int, default=2022)
    parser.add_argument('--training-seed', type=int, default=2022)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-inputs', action='store_true', help='Validate independent inputs; no model training')
    modes.add_argument('--prepare-only', action='store_true', help='Build and audit inner/refit inputs; no model training')
    return parser

def main():

    options = build_parser(('Data1', 'Data2', 'both'), 'both').parse_args()
    validate_options(options)
    datasets = ('Data1', 'Data2') if options.dataset == 'both' else (options.dataset,)
    if not options.check_inputs:
        for dataset in datasets:
            output = Path(options.output).resolve() / dataset
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'Use a fresh output directory: {output}')
    records = []
    for dataset in datasets:
        current = copy.copy(options)
        current.dataset = dataset
        result = run(current)
        records.append(result)
        print(result, flush=True)
    if not options.check_inputs and not options.prepare_only:
        output = Path(options.output).resolve()
        pd.DataFrame(records).to_csv(output / 'metrics_table.csv', index=False)
        pd.concat([pd.read_csv(output / dataset / 'outer_fold_metrics.csv') for dataset in datasets],
                  ignore_index=True).to_csv(output / 'metrics_by_fold.csv', index=False)
        from primary_metrics import PRIMARY_METRICS, write_table
        write_table([{'Dataset': record['dataset'],
                      **{label: record[label + '_mean'] for label in PRIMARY_METRICS}}
                     for record in records], output / 'primary_metrics.csv')

if __name__ == '__main__':
    main()
