from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

def embed_pca(x, dim=2):
    scaler = StandardScaler()
    pca    = PCA(n_components=dim)

    scaler.fit(x)
    x = scaler.transform(x)

    pca.fit(x)
    return pca.transform(x)

def separate(embeddings):
    x = []
    y = []
    label_map = []

    for agent in embeddings:
        if agent not in label_map:
            label_map.append(agent)

        for prompt in embeddings[agent]:
            x.append(prompt)
            y.append(label_map.index(agent))

    return np.array(x), np.array(y), label_map

def main():
    pool = PromptPool()

    pool.load(path="./data/prompts.jsonl",
              agent_path="./data/agents.jsonl")

    classifier = SBERTAgentRec("./models/test_model/")
    #classifier = SBERTAgentRec("all-mpnet-base-v2")
    classifier.fit(pool.pool)

    embeddings = classifier.embeddings
    x, y, label_map = separate(embeddings)

    x3d = embed_pca(x, dim=3)

    fig  = plt.figure()
    axis = fig.add_subplot(111, projection="3d")

    axis.scatter(x3d[:,0], x3d[:,1], x3d[:,2], c=y, cmap="plasma")
    fig.show()
    fig.savefig("test3d.png")
    plt.clf()

    x2d = embed_pca(x, dim=2)

    pc1, pc2 = zip(*x2d)
    plt.scatter(pc1, pc2, s=2, c=y, cmap="plasma")
    plt.show()
    plt.savefig("test2d.png")

if __name__ == "__main__":
    load_dotenv()
    main()
