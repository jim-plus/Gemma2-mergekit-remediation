import json
import os
import sys
from safetensors.torch import save_file
from safetensors import safe_open

configfile = "config.json"
indexfile = "model.safetensors.index.json"

f = open(configfile, 'r', encoding='utf-8')
config = json.load(f)
f.close()
arch = config.get("architectures")[0]
print("model architecture:",arch)
if (arch != "Gemma2ForCausalLM"):
    print("Gemma2 architecture not found; exiting")
    sys.exit()

f = open(indexfile, 'r', encoding='utf-8')
index = json.load(f)
f.close()
if "weight_map" in index and "lm_head.weight" in index["weight_map"]:
    print("excess lm_head.weight found in index; unnecessary in Gemma2 architecture")
else:
    print("no excess lm_head.weight found in index; exiting")
    sys.exit(0)
shardfile = index.get("weight_map")["lm_head.weight"]
del index["weight_map"]["lm_head.weight"]

print("shard:",shardfile)
f = safe_open(shardfile, framework="pt")
if ("lm_head.weight" in f.keys()):
    print("excess lm_head.weight found; unnecessary in Gemma2 architecture")
    print("removing weights from shard")
    tensors = { k:f.get_tensor(k) for k in f.keys() }
    del tensors["lm_head.weight"]
    save_file(tensors, shardfile+".fixed", metadata=f.metadata())
    os.replace(shardfile+".fixed", shardfile)
else:
    print("no excess lm_head.weight found in shard; we're good")

print("removing lm_head.weight entry from weight_map index")
with open(indexfile+".fixed", 'w', encoding='utf-8') as indexfix:
    json.dump(index, indexfix)
    indexfix.close()
    os.replace(indexfile+".fixed", indexfile)

