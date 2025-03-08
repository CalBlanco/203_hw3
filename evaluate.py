
from rouge import Rouge, FilesRouge
rouge = Rouge()

# the path of the gold
hyp_path = './data/test.txt.tgt'

# the path of the prediction
ref_path = './predictions.txt'

hypothesis = []
with open(hyp_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        hypothesis.append(l[:-1])

reference = []
with open(ref_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        reference.append(l[:-1])

rouge = Rouge()
print("Val", rouge.get_scores(hypothesis, reference, avg = True))
