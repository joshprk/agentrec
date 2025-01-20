from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

def main():
    pool = PromptPool()

    pool.load(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("all-mpnet-base-v2")
    classifier.fit(pool.pool)

    embeddings = classifier.embeddings
    visual_embeddings = {}

    for agent in embeddings:
        corpus = embeddings[agent]
        scaler = StandardScaler()
        pca    = PCA(n_components=3)

        scaler.fit(corpus)
        corpus = scaler.transform(corpus)

        pca.fit(corpus)
        visual_embeddings[agent] = pca.transform(corpus)

    X = []
    Y = []
    label_map = []

    for y in visual_embeddings:
        if y not in label_map:
            label_map.append(y)

        for x in visual_embeddings[agent]:
            X.append(x)
            Y.append(label_map.index(y))

    fig  = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    X = np.array(X)
    Y = np.array(Y)

    axis.scatter(X[:,0], X[:,1], X[:,2], c=Y, cmap="plasma")
    fig.show()

if __name__ == "__main__":
    load_dotenv()
    main()
