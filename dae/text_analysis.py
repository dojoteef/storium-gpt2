'''

    Analyze the linguistic

'''

from text_analysis_utils import get_diff_score

data = './judgement_contexts.json'


import json

with open(data) as f:
  data = json.load(f)

generated = data[0]['generated']['description']
finalized = data[1]['finalized']['description']


diff, diff_score = get_diff_score(generated, finalized)


print('Done')


