from datasets import load_dataset
import os

task_map={
    "sst2": "glue",
    "mrpc": "glue",
    "qqp":  "glue",
    "qnli": "glue",
    "mnli": "glue",
    "rte":  "glue",
    "stsb": "glue",
    "cola": "glue",
    "cb":   "super_glue",
    "wic":  "super_glue",
    "boolq": "super_glue", 
    "record": "super_glue",
    "wsc.fixed": "super_glue",
}

for k, v in task_map.items():
    dataset = load_dataset(v, k)
    dataset.save_to_disk(os.path.join("./data", v, k))

dataset = load_dataset("rajpurkar/squad")
dataset.save_to_disk(os.path.join("./data", "squad"))