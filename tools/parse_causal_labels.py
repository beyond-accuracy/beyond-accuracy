#!/usr/bin/env python3
"""Parse CausalAgents causal_labels.tfrecord and save it as a lookup dict.

proto layout (parsed manually):
  field 1 (repeated): causal agent ID set of each annotator (sub-message with repeated string field 1)
  field 2 (bytes):    scenario_id (hex string)

Output: unitraj/data/causal_labels_lookup.pkl
  {
    scenario_id: str → {
      'annotator_sets': List[List[str]],  # causal agent IDs from each of the 5 annotators
      'union':          Set[str],          # agents marked causal by at least one annotator
      'majority':       Set[str],          # agents marked causal by at least 3 of 5
    }
  }

Usage:
    # Run in a conda env with TensorFlow installed
    python tools/parse_causal_labels.py \
        --input /path/to/waymo_open_dataset_causal_agents/causal_labels.tfrecord \
        --output unitraj/data/causal_labels_lookup.pkl
"""
import argparse
import pickle
from collections import Counter
from pathlib import Path


def _parse_record(raw: bytes) -> tuple[str, list[list[str]]]:
    """Parse one TFRecord byte string and return (scenario_id, annotator_sets).

    Parses the proto wire format directly (works without causal_agents_pb2).
    """
    pos = 0
    annotator_sets: list[list[str]] = []
    scenario_id = ''

    while pos < len(raw):
        # varint tag
        tag_byte = raw[pos]; pos += 1
        field_num = tag_byte >> 3
        wire_type = tag_byte & 0x7

        if wire_type == 2:  # length-delimited
            # varint length
            length = 0; shift = 0
            while True:
                b = raw[pos]; pos += 1
                length |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            data = raw[pos: pos + length]; pos += length

            if field_num == 1:
                # sub-message: repeated field 1 = string agent IDs
                agent_ids: list[str] = []
                ipos = 0
                while ipos < len(data):
                    itag = data[ipos]; ipos += 1
                    if (itag >> 3) == 1 and (itag & 0x7) == 2:
                        ilen = data[ipos]; ipos += 1
                        agent_ids.append(data[ipos: ipos + ilen].decode())
                        ipos += ilen
                    else:
                        break  # unknown field, stop inner loop
                annotator_sets.append(agent_ids)

            elif field_num == 2:
                scenario_id = data.decode()
        else:
            # skip non-length-delimited fields (shouldn't occur here)
            pos += 1

    return scenario_id, annotator_sets


def build_lookup(tfrecord_path: str) -> dict:
    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    lookup: dict = {}

    for record in dataset:
        raw = record.numpy()
        scenario_id, annotator_sets = _parse_record(raw)
        if not scenario_id:
            continue

        counts: Counter = Counter()
        for s in annotator_sets:
            for aid in s:
                counts[aid] += 1

        union    = set(counts.keys())
        majority = {aid for aid, cnt in counts.items() if cnt >= 3}

        lookup[scenario_id] = {
            'annotator_sets': annotator_sets,
            'union':          union,
            'majority':       majority,
        }

    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='causal_labels.tfrecord of the Waymo CausalAgents release')
    parser.add_argument('--output', default='unitraj/data/causal_labels_lookup.pkl')
    args = parser.parse_args()

    print(f'Parsing: {args.input}')
    lookup = build_lookup(args.input)
    print(f'Parsed {len(lookup)} scenarios')

    has_causal = sum(1 for v in lookup.values() if v['union'])
    avg_union  = sum(len(v['union']) for v in lookup.values()) / max(len(lookup), 1)
    avg_maj    = sum(len(v['majority']) for v in lookup.values()) / max(len(lookup), 1)
    print(f'  With ≥1 causal agent : {has_causal} ({100*has_causal/len(lookup):.1f}%)')
    print(f'  Avg union set size   : {avg_union:.2f}')
    print(f'  Avg majority set size: {avg_maj:.2f}')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(lookup, f)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
