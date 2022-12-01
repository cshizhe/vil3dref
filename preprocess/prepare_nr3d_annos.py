import os
import argparse
import json
import jsonlines
import pandas as pd

from transformers import AutoTokenizer, AutoModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    data = pd.read_csv(args.input_file, header=0)
    print('process %d items' % (len(data)))

    num_reserved_items = 0
    with jsonlines.open(args.output_file, 'w') as outf:
        for i in range(len(data)):
            item = data.iloc[i]
            if not item['mentions_target_class']:
                continue
            enc_tokens = tokenizer.encode(item['utterance'])
            new_item = {
                'item_id': '%s_%06d' % (item['dataset'], i),
                'stimulus_id': item['stimulus_id'],
                'scan_id': item['scan_id'],
                'instance_type': item['instance_type'],
                'target_id': int(item['target_id']),
                'utterance': item['utterance'],
                'tokens': eval(item['tokens']),
                'enc_tokens': enc_tokens,
                'correct_guess': bool(item['correct_guess']),
            }
            if item['dataset'] == 'nr3d':
                new_item.update({
                    'uses_object_lang': bool(item['uses_object_lang']),
                    'uses_spatial_lang': bool(item['uses_spatial_lang']),
                    'uses_color_lang': bool(item['uses_color_lang']),
                    'uses_shape_lang': bool(item['uses_shape_lang'])
                })
            else:
                new_item.update({
                    'coarse_reference_type': item['coarse_reference_type'],
                    'reference_type': item['reference_type'],
                    'anchors_types': eval(item['anchors_types']),
                    'anchor_ids': eval(item['anchor_ids']),
                })
            # for k, v in new_item.items():
            #     print(k, type(v))
            outf.write(new_item)
            num_reserved_items += 1

    print('keep %d items' % (num_reserved_items))

if __name__ == '__main__':
    main()