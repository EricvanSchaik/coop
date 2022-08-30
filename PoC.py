from typing import List
import torch
from coop import VAE, util

import json
import pandas as pd
import rouge

model_name: str = "megagonlabs/bimeanvae-amzn"  # or "megagonlabs/bimeanvae-amzn", "megagonlabs/optimus-yelp", "megagonlabs/optimus-amzn"
vae = VAE(model_name)

reviews: List[str] = [
    "I love this ramen shop!! Highly recommended!!",
    "Here is one of my favorite ramen places! You must try!"
]
z_raw: torch.Tensor = vae.encode(reviews) # [num_reviews * latent_size]

# All combinations of input reviews
idxes: List[List[int]] = util.powerset(len(reviews))
# Taking averages for all combinations of latent vectors
zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]

outputs: List[str] = vae.generate(zs)

# Input-output overlap is measured by ROUGE-1 F1 score.
best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))

task = "amzn"  # or "amzn"
split = "dev"  # or "test"
data: List[dict] = json.load(open(f"./data/{task}/{split}.json"))
model_name: str = f"megagonlabs/bimeanvae-{task}"  # or f"megagonlabs/optimus-{task}"
vae = VAE(model_name)

hypothesis = []
for ins in data:
    reviews: List[str] = ins["reviews"]
    z_raw: torch.Tensor = vae.encode(reviews)
    idxes: List[List[int]] = util.powerset(len(reviews))
    zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]

    outputs: List[str] = vae.generate(zs, bad_words=util.BAD_WORDS)  # First-person pronoun blocking
    best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
    hypothesis.append(best)

reference: List[List[str]] = [ins["summary"] for ins in data]

evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                        stemming=True, ensure_compatibility=True)

scores = pd.DataFrame(evaluator.get_scores(hypothesis, reference))
print(scores)