"""Arms of the seeds study: depth and weight tying, at two learning rates. Same curriculum as the tab, smaller model."""
import json
base = {"V": 96, "T": 64, "d": 64, "H": 4, "B": 16, "mlp": 4, "ln": True, "lnAffine": False, "uniform": True, "tile": True, "pRep": 1, "rMin": 4, "rMax": 32, "probeB": 16, "probeT": 64}
shapes = [(2, 2), (2, 1), (3, 3), (3, 1), (4, 4), (4, 2), (4, 1)]
for lr, name in ((0.003, "study-lr3e-3"), (0.001, "study-lr1e-3")):
    arms = [{**base, "L": L, "unique": u, "lr": lr} for L, u in shapes]
    json.dump(arms, open(f"configs/search/{name}.json", "w"))
json.dump([{**base, "L": 1, "unique": 1, "lr": 0.003}, {**base, "L": 2, "unique": 2, "lr": 0.003, "pRep": 0}], open("configs/search/study-negatives.json", "w"))
deep = [{**base, "L": L, "unique": u, "lr": 0.003} for L, u in [(8, 8), (8, 2), (8, 1)]]
json.dump(deep, open("configs/search/study-deep.json", "w"))
