import csv
from typing import List
import torch
from coop import VAE, util
import pandas as pd
import json

dataset = "yelp"

model_name: str = "megagonlabs/bimeanvae-" + dataset  # or "megagonlabs/bimeanvae-amzn", "megagonlabs/optimus-yelp", "megagonlabs/optimus-amzn"
vae = VAE(model_name)


data = json.load(open("./data/" + dataset + "/products_8_reviews.json"))

hypothesis = list()
product_ids = list()
product_id = data[0]["product_id"]
reviews: List[str] = list()
for ins in data:
    if product_id == ins["product_id"]:
        reviews.append(ins["review_body"])
    else:
        z_raw: torch.Tensor = vae.encode(reviews)
        idxes: List[List[int]] = util.powerset(len(reviews))
        zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]

        outputs: List[str] = vae.generate(zs, bad_words=util.BAD_WORDS)  # First-person pronoun blocking
        best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
        hypothesis.append(best)

        product_ids.append(ins["product_id"])
        reviews = list()
    product_id = ins["product_id"]

result = pd.DataFrame(data={"product_id": product_ids, "text": hypothesis, "product_category": data[0]["product_category"]})
result.to_json("coop_on_" + dataset + "_summaries.json", orient="records")