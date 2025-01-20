from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from agentrec.datasets import PromptPool
from agentrec.models import SBERTAgentRec

import math

SHUFFLE_SEED = 42
TEST_SPLIT = 0.2
MODEL_ID = "./models/test_model/"
BASE_MODEL_ID = "all-mpnet-base-v2"
FILENAME_FORMAT = "./figures/test{dim}d.png"
PROMPT_PATH = "./data/test.jsonl"
AGENTS_PATH = "./data/agents.jsonl"

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

def plot3d(x, y, label_map, filename, s=2):
    fig  = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    pc0, pc1, pc2 = zip(*x)
    pc0, pc1, pc2 = np.array(pc0), np.array(pc1), np.array(pc2)

    for color, agent in enumerate(label_map):
        mask = y == color
        n = len(pc0[mask])
        plt.scatter(pc0[mask], pc1[mask], label=agent, s=s)

    axis.legend()
    fig.savefig(filename)
    fig.clear()

def plot2d(x, y, label_map, filename, s=2):
    pc0, pc1 = zip(*x)
    pc0, pc1 = np.array(pc0), np.array(pc1)

    for color, agent in enumerate(label_map):
        mask = y == color
        n = len(pc0[mask])
        plt.scatter(pc0[mask], pc1[mask], label=agent, s=s)

    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_tsne2d(x, y, label_map, filename, s=2):
    tsne = TSNE(n_components=2)
    x = tsne.fit_transform(x)
    x1, x2 = zip(*x)
    x1, x2 = np.array(x1), np.array(x2)

    for color, agent in enumerate(label_map):
        mask = y == color
        n =- len(x1[mask])
        plt.scatter(x1[mask], x2[mask], label=agent, s=s)

    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():
    pool = PromptPool()

    pool.load(path=PROMPT_PATH,
              agent_path=AGENTS_PATH)
    pool.shuffle(SHUFFLE_SEED)

    classifier = SBERTAgentRec(MODEL_ID)
    base_classifier = SBERTAgentRec(BASE_MODEL_ID)
    classifier.fit(pool.pool)
    base_classifier.fit(pool.pool)

    x, y, labels = separate(classifier.embeddings)
    base_x, base_y, base_labels = separate(base_classifier.embeddings)

    plot2d(embed_pca(x, dim=2), y, labels, "./figures/pcatest2d.png")
    plot3d(embed_pca(x, dim=3), y, labels, "./figures/pcatest3d.png")
    plot2d(embed_pca(base_x, dim=2), y, labels, "./figures/pcabase2d.png")
    plot3d(embed_pca(base_x, dim=3), y, labels, "./figures/pcabase3d.png")

    plot_tsne2d(base_x, y, labels, "./figures/tsnebase2d.png")
    plot_tsne2d(x, y, labels, "./figures/tsnetest2d.png")

if __name__ == "__main__":
    load_dotenv()
    main()
