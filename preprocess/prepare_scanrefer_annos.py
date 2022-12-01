import os
import argparse
import json
import jsonlines

from transformers import AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    num_reserved_items = 0
    with jsonlines.open(args.output_file, 'w') as outf:
        for split in ['train', 'val', 'test']:
            data = json.load(open(os.path.join(args.input_dir, 'ScanRefer_filtered_%s.json'%split)))
            print('process %s: %d items' % (split, len(data)))
            for i, item in enumerate(data):
                enc_tokens = tokenizer.encode(item['description'])
                outf.write({
                    'item_id': 'scanrefer_%s_%06d' % (split, i),
                    'scan_id': item['scene_id'],
                    'target_id': int(item['object_id']),
                    'instance_type': item['object_name'].replace('_', ' '),
                    'utterance': item['description'],
                    'tokens': item['token'],
                    'enc_tokens': enc_tokens,
                    'ann_id': item['ann_id']
                })
                num_reserved_items += 1

    print('keep %d items' % (num_reserved_items))

if __name__ == '__main__':
    main()