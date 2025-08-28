#!/usr/bin/env python3
'''
Generate meaningful session JSON for VSS demos.

- Produces 100k JSON lines into sessions.jsonl (change with --out/--n).
- Text is semantically structured (topics/subtopics/synonyms).
- Embeddings are deterministic and reflect text content (toy embedding).
- Matches the shape we will use for Redis Stack (RedisJSON + RediSearch + VECTOR).

Usage:
  python gen_sessions.py --n 100000 --out sessions.jsonl --dim 64 --seed 42
'''

import argparse
import base64
import json
import math
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict

from redis import Redis

import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configurable knobs (via CLI)
# ----------------------------

REDIS_URL = 'redis://localhost:6379/0'
em_model = SentenceTransformer('all-MiniLM-L6-v2')
SESSION_PREFIX = 'hk:session:'


def redis_client():
    return Redis.from_url(REDIS_URL, decode_responses=True)


def embed_sentence(sentence: str):
    '''Return a 384-dim semantic embedding for an English sentence.'''
    v = em_model.encode(sentence, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10, help='Number of sessions')
    p.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    return p.parse_args()


TOPICS: Dict[str, Dict[str, List[str]]] = {
    'yoga': {
        'subtopics': ['vinyasa', 'hatha', 'yin', 'power', 'restorative'],
        'syn': ['asanas', 'sun salutation', 'surya namaskar', 'flow', 'poses', 'stretching'],
        'hosts': ['AliceYoga', 'NishaFlow', 'RaviAsanas', 'MayaWellness', 'IshaVinyasa'],
        'tags': ['flexibility', 'balance', 'breath', 'mindful', 'mat', 'namaste'],
    },
    'meditation': {
        'subtopics': ['mindfulness', 'vipassana', 'guided', 'breathwork', 'loving-kindness'],
        'syn': ['awareness', 'calm', 'stillness', 'focus', 'mantra', 'zen'],
        'hosts': ['ZenWithKen', 'SanaMind', 'OmPrakash', 'LeenaCalm', 'TaraGuides'],
        'tags': ['mental-health', 'sleep', 'focus', 'relax', 'anxiety'],
    },
    'pilates': {
        'subtopics': ['reformer', 'mat', 'classical', 'contemporary', 'core'],
        'syn': ['controlled movement', 'stability', 'spine', 'core work', 'alignment'],
        'hosts': ['PaulaPilates', 'CoreKavya', 'GinaReformer', 'RituMat', 'AliaAlign'],
        'tags': ['core', 'posture', 'alignment', 'toning'],
    },
    'cardio': {
        'subtopics': ['hiit', 'aerobics', 'endurance', 'intervals', 'tabata'],
        'syn': ['heart rate', 'sweat', 'burpees', 'jumping jacks', 'conditioning'],
        'hosts': ['VikCardio', 'SamHIIT', 'NehaSweat', 'AriTabata', 'ElenaEndure'],
        'tags': ['fat-burn', 'endurance', 'sweat', 'metcon'],
    },
    'strength': {
        'subtopics': ['full-body', 'upper-body', 'lower-body', 'kettlebell', 'bodyweight'],
        'syn': ['compound lifts', 'hypertrophy', 'glutes', 'push pull', 'progressive overload'],
        'hosts': ['RohanLifts', 'MiraStrong', 'DevKBell', 'JasBodyweight', 'AshaHypertrophy'],
        'tags': ['muscle', 'strength', 'progression', 'toning'],
    },
    'dance': {
        'subtopics': ['zumba', 'bollywood', 'salsa', 'hip-hop', 'ballet-fit'],
        'syn': ['choreo', 'rhythm', 'groove', 'flow', 'music'],
        'hosts': ['ZoyaZumba', 'PriBollywood', 'LeoSalsa', 'HemaGroove', 'IkeHipHop'],
        'tags': ['fun', 'rhythm', 'cardio', 'group'],
    }
}

DIFFICULTY = ['beginner', 'intermediate', 'advanced', 'all-levels']
LANGUAGES = ['en', 'hi', 'es']  # English, Hindi, Spanish (example)
DURATIONS = [20, 30, 45, 60, 75, 90]  # minutes


def make_title(topic: str, sub: str) -> str:
    patterns = [
        f'{sub.title()} {topic.title()}',
        f'{topic.title()} for {sub.title()}',
        f'Morning {topic.title()} — {sub.title()}',
        f'{sub.title()} {topic.title()} Flow',
        f'{topic.title()} {sub.title()} Basics',
    ]
    return random.choice(patterns)

def make_description(topic: str, sub: str, syn: List[str], tags: List[str], duration: int, difficulty: str) -> str:
    bits = random.sample(syn, k=min(3, len(syn))) + random.sample(tags, k=min(2, len(tags)))
    return (
        f'A {duration}-minute {sub} {topic} session focused on '
        f"{', '.join(bits[:-1])} and {bits[-1]}. "
        f'Suitable for {difficulty} levels. Bring water and a mat.'
    )

def make_host_user(hosts: List[str]) -> List[Dict[str, str]]:
    # 1–2 hosts, occasionally 3
    k = 2 if random.random() < 0.85 else (3 if random.random() < 0.1 else 1)
    chosen = random.sample(hosts, k=min(k, len(hosts)))
    # Add a numeric suffix sometimes
    return [{'username': h if random.random() < 0.7 else f'{h}{random.randint(2,99)}'} for h in chosen]

def pick_times(now_ms: int) -> (int, int):
    '''
    Choose start/end over a ±45-day window.
    Bias lightly toward near-future and recent-past.
    '''
    # days offset: mixture of short and longer tails
    offset_days = int(np.random.choice(
        [-14,-10,-7,-3,-2,-1,0,1,2,3,7,10,14,21,30,45],
        p=np.array([0.05,0.02,0.05,0.06,0.07,0.09,0.10,0.10,0.10,0.09,0.07,0.05,0.04,0.04,0.02,0.05])
    ))
    start = datetime.utcfromtimestamp(now_ms/1000) + timedelta(days=offset_days, hours=random.randint(6,21))
    dur_min = random.choice(DURATIONS)
    end = start + timedelta(minutes=dur_min)
    return int(start.timestamp()*1000), int(end.timestamp()*1000)

def bool_biased(true_prob: float) -> bool:
    return random.random() < true_prob

def make_session(item_i) -> Dict:
    # choose a topic with weighted popularity
    topic_weights = {
        'yoga': 0.26, 'meditation': 0.19, 'pilates': 0.15,
        'cardio': 0.16, 'strength': 0.16, 'dance': 0.08
    }
    topics = list(topic_weights.keys())
    probs = np.array(list(topic_weights.values()), dtype=np.float64)
    probs /= probs.sum()
    topic = np.random.choice(topics, p=probs)

    sub = random.choice(TOPICS[topic]['subtopics'])
    title = make_title(topic, sub)
    # sprinkle extra context words for richer embeddings
    difficulty = random.choice(DIFFICULTY)
    duration = random.choice(DURATIONS)
    desc = make_description(topic, sub, TOPICS[topic]['syn'], TOPICS[topic]['tags'], duration, difficulty)
    lang = random.choice(LANGUAGES)

    # fuse a “vector passage” string from title + description + a few synonyms/tags
    extra = random.sample(TOPICS[topic]['syn'], k=min(2, len(TOPICS[topic]['syn']))) \
          + random.sample(TOPICS[topic]['tags'], k=min(2, len(TOPICS[topic]['tags'])))
    vector_passage_text = f'{title}. {desc} Keywords: ' + ', '.join(extra)
    embedding = embed_sentence(vector_passage_text)

    # schedule
    now_ms = int(time.time() * 1000)
    start_ms, end_ms = pick_times(now_ms)

    # flags & metadata
    interested = int(np.clip(np.random.normal(loc=14, scale=10), 0, 250))
    is_featured = bool_biased(0.12)
    is_paid = bool_biased(0.15)
    user_amount = 0 if not is_paid else random.choice([99, 149, 199, 299, 399])
    # recording mostly for past sessions, sometimes for live/upcoming as “replay available”
    has_recording = (end_ms < now_ms and bool_biased(0.7)) or (end_ms >= now_ms and bool_biased(0.2))
    watch_url = f'https://video.example.com/watch/{topic}/{sub}/{item_i+1}' if has_recording else ''

    # share flags (keep mostly public to match filters)
    is_shared_with_groups = bool_biased(0.03)

    hosts = make_host_user(TOPICS[topic]['hosts'])

    return {
        'session_id': str(item_i+1),
        'meta_data': {
            'status': 'published',
            'is_deleted': False,
            'blocked': False,
            'language': lang
        },
        'share_with': {
            'is_shared_with_groups': is_shared_with_groups
        },
        'schedule': {
            'start_time': start_ms,
            'end_time': end_ms,
            'duration_min': duration
        },
        'session_title': title,
        'topic': topic,
        'subtopic': sub,
        'difficulty': difficulty,
        'host_user': hosts,
        'user_engagement': {
            'interested_count': interested,
            'rating': round(float(np.clip(np.random.normal(4.3, 0.6), 2.5, 5.0)), 1),
            'ratings_count': int(np.clip(np.random.normal(30, 40), 0, 2000)),
        },
        'advance_option': {
            'feature': {'is_featured': is_featured},
            'user_amount': user_amount
        },
        'session_resources': {
            'watch_url': watch_url
        },
        'search_meta_data': {
            'vector_passage_text': vector_passage_text,
            'embedding': embedding.tolist()
        }
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    n = args.n

    r = redis_client()
    for i in range(n):
        doc = make_session(i)
        r.json().set(f'{SESSION_PREFIX}{i+1}', '$', doc)


if __name__ == '__main__':
    main()
