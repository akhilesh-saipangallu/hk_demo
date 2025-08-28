import argparse
import math
import sys
import time
import json
import re
import time

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag, Num, Text
from redis.commands.search.aggregation import Desc

import numpy as np
from sentence_transformers import SentenceTransformer


REDIS_URL = 'redis://localhost:6379/0'
em_model = SentenceTransformer('all-MiniLM-L6-v2')

# prefixes
SESSION_PREFIX = 'hk:session:'
# indexes
SESSION_INDEX = 'hk:idx:session'

WORD_RE = re.compile(r'\w+', re.UNICODE)

GLOBAL_REDIS = Redis.from_url(REDIS_URL, decode_responses=True)

def redis_client():
    return GLOBAL_REDIS
    # return Redis.from_url(REDIS_URL, decode_responses=True)


def print_with_separator(tag, *args):
    s, x = '=', 15
    print(s*x, tag, s*x)
    print(*args)
    print(s*x, tag, s*x)


def embed_sentence(sentence: str):
    '''Return a 384-dim semantic embedding for an English sentence.'''
    v = em_model.encode(sentence, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def session_index_schema():
    return IndexSchema.from_dict({
        'index': {
            'name': SESSION_INDEX,
            'prefix': SESSION_PREFIX,
            'storage_type': 'json',
        },
        'fields': [
            {'name': 'session_id', 'path': '$.session_id', 'type': 'tag'},
            {'name': 'status', 'path': '$.meta_data.status', 'type': 'tag'},
            {'name': 'is_deleted', 'path': '$.meta_data.is_deleted', 'type': 'tag'},
            {'name': 'blocked', 'path': '$.meta_data.blocked', 'type': 'tag'},
            {'name': 'is_shared_with_groups', 'path': '$.share_with.is_shared_with_groups', 'type': 'tag'},
            {'name': 'start_time', 'path': '$.schedule.start_time', 'type': 'numeric', 'attrs': {'sortable': True}},
            {'name': 'end_time', 'path': '$.schedule.end_time', 'type': 'numeric', 'attrs': {'sortable': True}},
            {'name': 'interested_count', 'path': '$.user_engagement.interested_count', 'type': 'numeric', 'attrs': {'sortable': False}},
            {'name': 'is_featured', 'path': '$.advance_option.feature.is_featured', 'type': 'tag'},
            {'name': 'user_amount', 'path': '$.advance_option.user_amount', 'type': 'numeric', 'attrs': {'sortable': False}},
            {'name': 'watch_url', 'path': '$.session_resources.watch_url', 'type': 'text', 'attrs': {'sortable': False}},
            {'name': 'session_title', 'path': '$.session_title', 'type': 'text', 'attrs': {'sortable': False}},
            {'name': 'host_username', 'path': '$.host_user[*].username', 'type': 'text', 'attrs': {'sortable': False}},
            {'name': 'passage', 'path': '$.search_meta_data.vector_passage_text', 'type': 'text', 'attrs': {'sortable': False}},
            {
                'name': 'embedding',
                'path': '$.search_meta_data.embedding',
                'type': 'vector',
                'attrs': {
                    'algorithm': 'HNSW', # or 'FLAT'
                    'datatype': 'FLOAT32',
                    'dims': 384,
                    'distance_metric': 'COSINE',
                }
            }
        ],
    })


def action_init(**kwargs):
    r = redis_client()

    # Articles index
    art_idx = SearchIndex(session_index_schema(), r)
    if not art_idx.exists():
        art_idx.create()
        print(f'Created index: {SESSION_INDEX}')
    else:
        print(f'Index exists: {SESSION_INDEX}')


def action_search(**kwargs):
    '''
    Kwargs:
        search_text: search text
    '''

    search_text = kwargs.get('search_text')
    if not search_text:
        print_with_separator('error', 'provide search_text')
        return

    knn_k = kwargs.get('knn_k')
    if not knn_k:
        knn_k = 110

    fts_k = kwargs.get('fts_k')
    if not fts_k:
        fts_k = 10

    limit = kwargs.get('limit')
    if not limit:
        limit = 10

    skip_stats = False
    iter = kwargs.get('iter')
    if not iter:
        iter = 1
        skip_stats = True

    times = []
    for i in range(iter):
        start_time = time.perf_counter()
        vec_rows = vector_branch(search_text, k=knn_k)
        fts_rows = text_branch(search_text, k=fts_k)
        now_ms = time.time() * 1000
        final_res = union_and_rank(vec_rows, fts_rows, search_text, now_ms, limit)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    if not skip_stats:
        times = np.array(times)
        p90 = np.percentile(times, 90)
        print_with_separator('p90', p90)

    if skip_stats:
        print(json.dumps(final_res, indent=4))


def get_base_filter():
    return (
        (Tag('status') == 'published') &
        (Tag('is_deleted') == 'false') &
        (Tag('blocked') == 'false') &
        (Tag('is_shared_with_groups') == 'false')
    )


def vector_branch(search_text, k=110):
    search_emb = embed_sentence(search_text)

    idx = SearchIndex(session_index_schema(), redis_client())
    res = idx.query(
        VectorQuery(
            vector=search_emb,
            vector_field_name='embedding',
            num_results=k,
            return_fields=[
                'session_title',
                'host_username',
                'start_time',
                'end_time',
                'watch_url',
                'interested_count',
                'is_featured',
                'user_amount',
                'vector_distance'
            ],
            return_score=True,
            filter_expression=get_base_filter(),
            dtype='float32',
            sort_by='vector_distance',
        ).sort_by('vector_distance', asc=True).dialect(2)
    )

    out = []
    for i, doc in enumerate(res):
        doc['key'] = doc['id']
        doc['vs_score'] = round(1.0 / (i + 60), 6)
        out.append(doc)

    return out


def text_branch(search_text, k=10):
    """
    Runs BM25 full-text over title + host usernames with autocomplete-like prefixes.
    Returns docs ordered by BM25, with rank-based fts_score.
    """

    filter_expression = get_base_filter()
    filter_expression = filter_expression & ((Text('session_title') % f'{search_text}*') | (Text('host_username') % f'{search_text}*'))

    idx = SearchIndex(session_index_schema(), redis_client())
    res = idx.query(
        FilterQuery(
            return_fields=[
                'session_title',
                'host_username',
                'start_time',
                'end_time',
                'watch_url',
                'interested_count',
                'is_featured',
                'user_amount',
            ],
            filter_expression=filter_expression,
            num_results=k,
        ).dialect(2)
    )

    out = []
    for i, doc in enumerate(res):
        doc['key'] = doc['id']
        doc['fts_score'] = round(1.0 / (i + 60), 6)
        out.append(doc)

    return out


def is_nonempty(s) -> bool:
    return bool(s) and s not in ('None', 'null', 'NULL')


def jaccard_similarity(query_text: str, title: str) -> float:
    q_tokens = set(t.lower() for t in WORD_RE.findall(query_text))
    t_tokens = set(t.lower() for t in WORD_RE.findall(title or ''))
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def compute_bucket_and_score(row, query_text, now_ms):
    '''
    Adds: exact_match, jaccard, is_live/upcoming/past, has_recording, group, composite, norm.
    '''

    title = row.get('session_title', '') or ''
    host = row.get('host_username', '') or ''
    qlower = (query_text or '').lower()

    exact = int(qlower in title.lower() or qlower in host.lower())
    jacc = jaccard_similarity(query_text, title)

    start = int(row.get('start_time') or 0)
    end = int(row.get('end_time') or 0)
    is_live = int(start <= now_ms <= end)
    is_upcoming = int(start > now_ms)
    is_past = int(end < now_ms)
    has_rec = int(is_nonempty(row.get('watch_url')))

    # Bucket rules (mirror the Mongo $switch order)
    if exact >= 1 and is_live:
        group = 1
    elif exact >= 1 and is_upcoming:
        group = 2
    elif exact >= 1 and is_past and has_rec:
        group = 3
    elif exact >= 1 and is_past and not has_rec:
        group = 4
    elif jacc >= 0.3 and is_live:
        group = 5
    elif jacc >= 0.3 and is_upcoming:
        group = 6
    elif jacc >= 0.3 and has_rec:
        group = 7
    else:
        group = 7

    vs_score = float(row.get('vs_score', 0))
    fts_score = float(row.get('fts_score', 0))

    composite = (
        exact * 1000 +
        is_live * 300 +
        is_upcoming * 500 +
        jacc * 500 +
        has_rec * 200 +
        fts_score + vs_score
    )

    row.update({
        'exact_match': exact,
        'jaccard': round(jacc, 6),
        'is_live': is_live,
        'is_upcoming': is_upcoming,
        'is_past': is_past,
        'has_recording': has_rec,
        'group': group,
        'composite': round(composite, 6),
        'normalized_score': round(composite / 2000.0, 6),
    })

def union_and_rank(vec_rows, fts_rows, query_text, now_ms, limit):
    '''
    Merge by key, keep max vs_score/fts_score per id, add features, bucket, composite; sort and slice.
    '''

    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    by_key = {}

    # Insert vector rows
    for r0 in vec_rows:
        key = r0['key']
        cur = by_key.get(key)
        if not cur:
            by_key[key] = dict(r0)
        else:
            # keep best of each score
            cur['vs_score'] = max(float(cur.get('vs_score', 0)), float(r0.get('vs_score', 0)))
            # merge other fields conservatively
            for k, v in r0.items():
                if k not in cur or not cur[k]:
                    cur[k] = v

    # Insert FTS rows
    for r0 in fts_rows:
        key = r0['key']
        cur = by_key.get(key)
        if not cur:
            by_key[key] = dict(r0)
        else:
            cur['fts_score'] = max(float(cur.get('fts_score', 0)), float(r0.get('fts_score', 0)))
            for k, v in r0.items():
                if k not in cur or not cur[k]:
                    cur[k] = v

    # Fill missing score fields with zeros
    for row in by_key.values():
        row.setdefault('vs_score', 0.0)
        row.setdefault('fts_score', 0.0)
        compute_bucket_and_score(row, query_text, now_ms)

    rows = list(by_key.values())
    rows.sort(key=lambda r: (r['group'], -r['composite']))
    return rows[:limit]


def parse_kv_args(kv_list):
    out = {}
    for item in kv_list:
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        # naive cast to int/float if possible
        if v.isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action',
        choices=['init', 'search'],
        help='Action to run'
    )
    parser.add_argument('params', nargs='*', help='Optional key=value params for the action')
    args = parser.parse_args()

    kwargs = parse_kv_args(args.params)

    if args.action == 'init':
        action_init(**kwargs)
    elif args.action == 'search':
        action_search(**kwargs)
    else:
        print('unknown action', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
