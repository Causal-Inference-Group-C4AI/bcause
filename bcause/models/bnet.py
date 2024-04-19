from typing import Union, Iterable

import networkx as nx

from bcause.factors.mulitnomial import random_multinomial, uniform_multinomial
from bcause.inference.probabilistic.elimination import VariableElimination
from bcause.models.pgmodel import DiscreteDAGModel
from bcause.util.domainutils import subdomain
from bcause.util.graphutils import dag2str


class BayesianNetwork(DiscreteDAGModel):

    def __init__(self, dag:Union[nx.DiGraph,str], factors:Union[dict, Iterable] = None, check_factors:bool = True):

        self._initialize(dag)
        self._check_factors = check_factors
        if factors is not None: self._set_factors(factors)

    from bcause.readwrite import bnwrite, bnread
    _reader, _writer = bnread, bnwrite

    def builder(self, **kwargs):
        return BayesianNetwork(**kwargs)


    def __repr__(self):
        str_card = ",".join([f"{str(v)}:{'' if d is None else str(len(d))}" for v, d in self._domains.items()])

        return f"<BayesianNetwork ({str_card}), dag={dag2str(self.graph)}>"

    def randomize_factors(self, domains:dict):
        for v in self.variables:
            parents = self.get_parents(v)
            dom = {x: d for x, d in domains.items() if x == v or x in parents}
            self.set_factor(v, random_multinomial(dom, right_vars=parents))

    @staticmethod
    def __buid(dag, domains, factor_builder):
        bn = BayesianNetwork(dag)
        v = "x"

        bn.get_parents(v)
        for v in bn.variables:
            dom_v = subdomain(domains, v, *bn.get_parents(v))
            fv = factor_builder(dom_v, bn.get_parents(v))
            bn.set_factor(v, fv)

        return bn
    @staticmethod
    def buid_uniform(dag, domains):
        return BayesianNetwork.__buid(dag, domains, uniform_multinomial)
    def buid_random(dag, domains):
        return BayesianNetwork.__buid(dag, domains, random_multinomial)


    def arc_inversion(self, arc):
        x, y = arc[0], arc[1]

        # check that arc exists
        if not arc in self.graph.edges:
            raise ValueError(f"Arc {arc} does not exists in DAG.")

        if len(self.get_parents(arc[1])) > 1:
            raise NotImplementedError("Operation not implementing for node with other parents.")
        else:
            new_dag = nx.DiGraph([e for e in self.graph.edges if e != arc] + [(y, x)])
            new_factors = dict()

            inf = VariableElimination(self)

            for v in self.variables:
                if v not in arc:
                    new_factors[v] = self.get_factors(v)[0]
                else:
                    if v == x:
                        pa = self.get_parents(v) + [y]
                    else:
                        pa = self.get_parents(v).remove(x)

                    new_factors[v] = inf.query(v, conditioning=pa)

            return self.builder(dag=new_dag, factors=new_factors)


if __name__ == "__main__":

    dag = nx.DiGraph([("A","B"), ("C","B"), ("C","D")])
    domains = dict(A=["a1", "a2"], B=[0, 1, 3], C=["c1", "c2", "c3", "c4"], D=[True, False])
    factors = dict()

    for v in dag.nodes:
        parents = list(dag.predecessors(v))
        dom = {x:d for x,d in domains.items() if x == v or x in parents}
        factors[v]  = random_multinomial(dom, right_vars=parents)

    bnet = BayesianNetwork(dag, factors)

    bnet = BayesianNetwork("[A][B|A:C][C][D|C]", factors)

    bnet = BayesianNetwork("[A][B|A:C][C][D|C]")

    bnet.randomize_factors(domains)
    print(bnet.get_domains(bnet.variables))

