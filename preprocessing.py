import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from utils_direction import (DATASETS,read_matrix,read_json,write_json,sha256,set_random_seed)

EMPTY_CFS_RULE='CFS is zero if either circRNA has no training-associated diseases, including its diagonal'

def array_hash(value):

    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()

def validate_pairs(pairs, shape, name='pairs'):

    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 3 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f'{name}: expected integer N x 3 samples')
    if not len(pairs) or not np.isin(pairs[:, 2], [0, 1, 2]).all():
        raise ValueError(f'{name}: empty samples or invalid labels')
    if (pairs[:, :2] < 0).any() or (pairs[:, 0] >= shape[0]).any() or (pairs[:, 1] >= shape[1]).any():
        raise ValueError(f'{name}: node index out of range')
    if len(np.unique(pairs[:, :2], axis=0)) != len(pairs):
        raise ValueError(f'{name}: duplicate/conflicting pairs')
    return pairs

def assert_disjoint(train, heldout, shape):

    validate_pairs(train, shape, 'training')
    validate_pairs(heldout, shape, 'heldout')
    train_ids = train[:, 0] * shape[1] + train[:, 1]
    heldout_ids = heldout[:, 0] * shape[1] + heldout[:, 1]
    if np.intersect1d(train_ids, heldout_ids).size:
        raise ValueError('Training/heldout pair overlap, including unknown samples')

def supervised_relations(train, shape):

    train = validate_pairs(train, shape, 'stage training')
    up = np.zeros(shape, dtype=np.float32)
    down = np.zeros(shape, dtype=np.float32)
    for label, matrix in ((1, up), (2, down)):
        pairs = train[train[:, 2] == label]
        matrix[pairs[:, 0], pairs[:, 1]] = 1
    return up + down, up, down

def validate_dss(dss, disease_count):

    dss = np.asarray(dss, dtype=np.float64)
    if dss.shape != (disease_count, disease_count) or not np.isfinite(dss).all():
        raise ValueError('DSS shape/values invalid')
    if not np.allclose(dss, dss.T, atol=1e-8, rtol=0):
        raise ValueError('DSS must be symmetric in the sample disease order')
    if dss.min() < -1e-6:
        raise ValueError('DSS contains negative values beyond numerical tolerance')
    detail = {'dss_negative_roundoff_entries_set_to_zero': int((dss < 0).sum()),
              'dss_source_values_sha256': array_hash(dss),
              'dss_upper_values_clipped': False}
    dss = np.maximum(dss, 0)
    detail['dss_used_sha256'] = array_hash(dss)
    return dss, detail

def functional_similarity(a_train, dss):


    a = np.asarray(a_train)
    if a.ndim != 2 or not np.isin(a, [0, 1]).all():
        raise ValueError('CFS requires a binary training association matrix')
    dss, _ = validate_dss(dss, a.shape[1])
    best = np.zeros(a.shape, dtype=np.float64)
    for row, profile in enumerate(a):
        diseases = np.flatnonzero(profile)
        if len(diseases):
            best[row] = dss[:, diseases].max(axis=1)
    forward = np.asarray(csr_matrix(a, dtype=np.float64) @ best.T)
    counts = a.sum(axis=1, dtype=np.float64)
    denominator = counts[:, None] + counts[None, :]
    cfs = np.divide(forward + forward.T, denominator,
                    out=np.zeros_like(forward), where=denominator > 0)
    empty = counts == 0
    cfs[empty, :] = 0
    cfs[:, empty] = 0
    return cfs

def gip(profiles):

    profiles = np.asarray(profiles, dtype=np.float64)
    squared_norm = np.einsum('ij,ij->i', profiles, profiles)
    mean_norm = float(squared_norm.mean())
    gamma = 1 / mean_norm if mean_norm > 0 else 1.
    distance = squared_norm[:, None] + squared_norm[None, :] - 2 * profiles @ profiles.T
    np.maximum(distance, 0, out=distance)
    result = np.exp(-gamma * distance).astype(np.float32)
    np.fill_diagonal(result, 1)
    return result, gamma

def check_heldout_edges(folder, heldout):

    with np.load(Path(folder) / 'stage_relations.npz', allow_pickle=False) as matrices:
        for key in ('A_train', 'A_up', 'A_down'):
            if matrices[key][heldout[:, 0], heldout[:, 1]].any():
                raise AssertionError(f'Heldout edge entered stage relation {key}')
def relations_from_training(train, shape, background):

    train = validate_pairs(train, shape)
    if background.shape != shape or not np.isin(background, [0, 1]).all():
        raise ValueError('Invalid background associations')
    if background[train[:, 0], train[:, 1]].any():
        raise ValueError('Background overlaps supervised pairs')
    _, up, down = supervised_relations(train, shape)
    return np.maximum(background, up + down), up, down

def stage_similarities(a_train, dss):

    dss, detail = validate_dss(dss, a_train.shape[1])
    cfs = functional_similarity(a_train, dss)
    circ_gip, circ_gamma = gip(a_train)
    disease_gip, disease_gamma = gip(a_train.T)
    degree = a_train.sum(axis=0, dtype=np.float64)
    rho = degree / degree.max() if degree.max() > 0 else np.zeros_like(degree)
    disease_weight = (rho[:, None] + rho[None, :]) / 2
    delta = np.abs(cfs - circ_gip)
    omega = np.exp(-delta)
    circ = omega * (cfs + circ_gip) / 2 + (1 - omega) * np.minimum(cfs, circ_gip)
    disease = disease_weight * disease_gip + (1 - disease_weight) * dss
    matrices = {'CFS': cfs, 'DSS': dss, 'circ_GIP': circ_gip, 'disease_GIP': disease_gip,
        'disease_normalized_degree': rho, 'disease_GIP_weight': disease_weight,
        'circ_consistency_weight': omega, 'circ_similarity': circ.astype(np.float32),
        'dis_similarity': disease.astype(np.float32)}
    if not all(np.isfinite(value).all() for value in matrices.values()):
        raise ValueError('Nonfinite stage similarities')
    audit = {**detail, 'A_train_sha256': array_hash(a_train),
        'cfs_and_gip_recomputed_from_stage_training': True,
        'cfs_empty_set_rule': EMPTY_CFS_RULE,
        'circ_gip_gamma': circ_gamma, 'disease_gip_gamma': disease_gamma,
        'circ_fusion': 'Eq. (3)-(4): exp(-abs(CFS-GIP))*mean + (1-weight)*minimum',
        'disease_fusion': 'Eq. (2): pair mean of degree/max_degree weights GIP versus DSS',
        'similarity_diagonal_overwritten': False,
        'similarity_fingerprints': {key: array_hash(value) for key, value in matrices.items()}}
    return matrices, audit

def row_normalize(values):

    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-8)

def prepare_stage(args, device, train, shape, cm, md, dss, background, seed, folder):

    from sklearn.decomposition import PCA
    set_random_seed(seed, deterministic=True)
    a, up, down = relations_from_training(train, shape, background)
    matrices, audit = stage_similarities(a, dss)
    circ, disease = matrices['circ_similarity'], matrices['dis_similarity']
    bridge = row_normalize(cm @ md)
    circ_input = np.concatenate((circ, cm, bridge), axis=1)
    disease_input = np.concatenate((disease, md.T, bridge.T), axis=1)
    args.hyper_pca_dim = min(128, *circ_input.shape, *disease_input.shape)
    circ_pca = PCA(n_components=args.hyper_pca_dim).fit_transform(circ_input)
    disease_pca = PCA(n_components=args.hyper_pca_dim).fit_transform(disease_input)
    hc, hd = {}, {}
    for branch, relation in (('all', a), ('up', up), ('down', down)):
        hc[branch] = torch.as_tensor(np.concatenate((row_normalize(relation), row_normalize(cm),
            .5 * row_normalize(bridge)), axis=1), dtype=torch.float32, device=device)
        hd[branch] = torch.as_tensor(np.concatenate((row_normalize(relation.T), row_normalize(md.T),
            .5 * row_normalize(bridge.T)), axis=1), dtype=torch.float32, device=device)
    prepared = (torch.as_tensor(circ_pca, dtype=torch.float32, device=device),
                torch.as_tensor(disease_pca, dtype=torch.float32, device=device), hc, hd, circ, disease)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'stage_relations.npz', A_train=a, A_up=up, A_down=down, background=background)
    np.savez_compressed(folder / 'stage_similarities.npz', **matrices)
    fingerprints, degrees = {}, {}
    for name, value in zip(('circ_pca', 'disease_pca', 'H_circ', 'H_dis'), prepared):
        values = value if isinstance(value, dict) else {'features': value}
        fingerprints[name] = {}
        for branch, tensor in values.items():
            array = tensor.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite preprocessing: {name}/{branch}')
            fingerprints[name][branch] = array_hash(array)
            if name.startswith('H_'):
                degrees[f'{name}_{branch}_node_degree'] = array.sum(axis=1)
                degrees[f'{name}_{branch}_edge_degree'] = array.sum(axis=0)
    np.savez_compressed(folder / 'stage_graph_degrees.npz', **degrees)
    audit.update({'training_pairs_sha256': array_hash(train), 'all_edges': int(a.sum()),
        'training_up_edges': int(up.sum()), 'training_down_edges': int(down.sum()),
        'background_edges': int(background.sum()), 'background_sha256': array_hash(background),
        'pca_effective_dim': args.hyper_pca_dim, 'pca_rebuilt_for_this_stage': True,
        'preprocessing_seed': seed, 'graphs_and_degree_normalization_rebuilt_for_this_stage': True,
        'relation_block_weights': {'all': 1., 'up': 1., 'down': 1., 'miRNA': 1., 'bridge': .5},
        'preprocessed_fingerprints': fingerprints,
        'protocol': 'pairwise transductive; heldout target pairs excluded; permitted background retained'})
    write_json(folder / 'feature_audit.json', audit)
    return prepared

def load_independent_inputs(dataset, data_root, source_root):

    spec = DATASETS[dataset]
    folder = Path(data_root).resolve() / 'original_data' / dataset
    dss_path = folder / 'DSS.npy'
    raw_dss = np.load(dss_path, allow_pickle=False)
    dss, detail = validate_dss(raw_dss, spec['shape'][1])
    files = [dss_path]
    manifest_path = folder / 'data_manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['dataset'] != dataset:
            raise ValueError('DSS manifest dataset mismatch')
        if sha256(dss_path) != manifest['components']['disease']['outputs']['DSS.npy']:
            raise ValueError('DSS source hash mismatch')
        files.append(manifest_path)
    raw = Path(source_root).resolve()
    cm_path, md_path = raw / spec['a_cm'], raw / spec['a_md']
    cm = read_matrix(cm_path, (spec['shape'][0], spec['mirnas']))
    md = read_matrix(md_path, (spec['mirnas'], spec['shape'][1]))
    if not np.isin(cm, [0, 1]).all() or not np.isin(md, [0, 1]).all():
        raise ValueError('The manuscript defines binary auxiliary relations')
    files.extend((cm_path, md_path))
    metadata = {**detail, 'input_files': [str(path.resolve()) for path in files],
                'DSS_independent_disease_ontology': True,
                'A_cm_A_md_independent_of_heldout_targets': True,
                'provenance_basis': 'User-confirmed input sources',
                'node_order': 'DSS disease order and auxiliary node order match original step5 indices'}
    return dss, cm, md, files, metadata

def load_background_associations(dataset, source_root, data_root):

    spec = DATASETS[dataset]
    raw = Path(source_root).resolve()
    path = raw / f'Association Matrix_{spec["suffix"]}.csv'
    original = read_matrix(path, spec['shape'])
    if not np.isin(original, [0, 1]).all():
        raise ValueError('Original association matrix must be binary')
    background = original.astype(np.float32)
    source = Path(data_root) / f'step5_training_samples_{spec["suffix"]}'
    coordinates = []
    files = [path]
    for fold in range(5):
        file = source / f'fold_{fold}' / 'test_pairs.npy'
        pairs = np.load(file, mmap_mode='r', allow_pickle=False)
        if pairs.ndim != 2 or pairs.shape[1] != 3 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError(f'Invalid pair array: {file}')
        coords = np.array(pairs[:, :2], copy=True)
        if (coords < 0).any() or (coords[:, 0] >= spec['shape'][0]).any() or (coords[:, 1] >= spec['shape'][1]).any():
            raise ValueError(f'Invalid pair coordinates: {file}')
        coordinates.append(coords)
        files.append(file)
    coordinates = np.concatenate(coordinates)
    ids = coordinates[:, 0] * spec['shape'][1] + coordinates[:, 1]
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Overlapping outer test folds')
    background[coordinates[:, 0], coordinates[:, 1]] = 0
    metadata = {'original_association_edges': int(original.sum()),
        'retained_background_edges': int(background.sum()),
        'all_supervised_pair_coordinates_removed': True,
        'heldout_direction_labels_used': False,
        'background_sha256': array_hash(background)}
    return background, files, metadata
