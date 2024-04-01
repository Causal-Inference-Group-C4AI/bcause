import networkx as nx
from networkx import bfs_tree

from bcause import BayesianNetwork, MultinomialFactor
from bcause.factors.mulitnomial import uniform_multinomial
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.learning.parameter import MaximumLikelihoodEstimation
from bcause.models.classification.classifier import  BNetClassifier
from bcause.util.domainutils import subdomain
from bcause.util.probability import mutual_info_data


class TAN(BNetClassifier):
    def __init__(self, domains, classVar, root = None):
        self._root = root
        super().__init__(domains,classVar)

    def _build_model(self) -> BayesianNetwork:

        # Build undirected max-SP graph
        G = nx.Graph()
        for i in range(len(self._attributes) - 1):
            for j in range(i + 1, len(self._attributes)):
                x, y = self._attributes[i], self._attributes[j],
                info = mutual_info_data(data, x, y, self._classvar, self._domains)
                G.add_edge(x, y, weight=info)

        undirected_stree = nx.maximum_spanning_tree(G)

        # transform it into a directed tree
        dag = bfs_tree(undirected_stree, self._root or self._attributes[0])

        # Add the edges from the class
        for v in self._attributes:
            dag.add_edge(self._classvar, v)

        # Build the BN object
        return BayesianNetwork.buid_uniform(dag, self._domains)


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("./data/igd_test.csv", index_col=0)
    classVar = "y"
    domains = {c:list(data[c].unique()) for c in data.columns}


    clf = TAN(domains, classVar, root="s")
    clf.fit(data)

    y_pred = clf.predict(data)

    clf.model.factors

    clf.predict(data.loc[0:10])
    clf.predict_proba(data.loc[0:10])

    i = 1
    for i in range(0,10):
        print(f"{data.loc[[i]]} {clf.predict_proba(data.loc[[i]])}")

    y = list(data["y"])

    clf.model.domains

    print(sum(y == y_pred))
    clf.model


