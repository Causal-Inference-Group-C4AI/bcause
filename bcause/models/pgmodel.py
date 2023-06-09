from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union, Dict, Hashable

import networkx as nx
import pandas as pd
from networkx import relabel_nodes

import bcause.factors.factor as bf
import bcause.util.graphutils as gutils

from bcause.models.sampling import forward_sampling

class PGModel(ABC):

    _reader, _writer = None,None

    def _initialize(self, graph:nx.Graph):
        self._graph = graph
        self._factors = {x: None for x in graph.nodes}
        self._domains = {x: None for x in graph.nodes}

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def variables(self) -> list:
        return list(self._graph.nodes)

    @property
    def factors(self) -> Union[Dict,list]:
        return self._factors

    def get_factors(self, *variables) -> list:
        return [self._factors[v] for v in variables]


    def set_factor(self, var:Hashable, f:bf.DiscreteFactor):
        # check type
        if not isinstance(f, bf.DiscreteFactor):
            raise ValueError(f"Factor {var} must  be discrete")
        if not isinstance(f, bf.ConditionalFactor): raise ValueError(f"Factor {var} must  be conditional")

        # check left right variables
        if set(self.get_parents(var)) != set(f.right_vars): raise ValueError(f"Wrong right variables in factor for {var}")
        if set([var]) != set(f.left_vars):
            raise ValueError("Wrong left variables in factor")

        # check domains
        for v in f.variables:
            if v in self._domains and self._domains[v] != None:
                    if set(self._domains[v]) != set(f.domain[v]): raise ValueError(f"Inconsistent domain for {v} when setting {var}")

        # Update domains
        if var not in self._domains or self._domains[var] == None:
            self._domains[var] = f.domain[var]

        # update factor dictionary
        self._factors[var] = f

    def save(self, path):
        if str(path).endswith(".uai"):
            self._writer.to_uai(model=self,path=path)
        elif str(path).endswith(".xmlbif"):
            self._writer.to_xmlbif(model=self,path=path)
        elif str(path).endswith(".bif"):
            self._writer.to_bif(model=self,path=path)
        else:
            raise ValueError(f"Unknown format for {path}")


    @classmethod
    def read(cls, path):
        if str(path).endswith(".uai"):
            return cls._reader.from_uai(path=path)
        elif str(path).endswith(".xmlbif"):
            return cls._reader.from_xmlbif(path=path)
        elif str(path).endswith(".bif"):
            return cls._reader.from_xmlbif(path=path)
        else:
            raise ValueError(f"Unknown format for {path}")


class DiscreteDAGModel(PGModel):

    def _initialize(self, dag:Union[nx.Graph, str]):
        if isinstance(dag, str):
            dag = gutils.str2dag(dag)
        if not isinstance(dag, nx.DiGraph) or len(list(nx.simple_cycles(dag)))>0:
            raise ValueError("Input graph must be a DAG")
        super()._initialize(dag)

    @abstractmethod
    def builder(self, **kwargs):
        pass

    def _set_factors(self, factors):
        if isinstance(factors, dict):
            for v,f in factors.items():
                self.set_factor(v,f)
        elif isinstance(factors, Iterable):
            for f in factors:
                if len(f.left_vars)>1: raise ValueError("Factor with more than one left variable.")
                self.set_factor(f.left_vars[0], f)
        else:
            raise ValueError("Wrong factor type")


    def get_domains(self, variables):
        return {v:d for v,d in self._domains.items() if v in variables}

    def get_varsizes(self, variables):
        return {v:len(d) for v,d in self._domains.items() if v in variables}


    @property
    def domains(self):
        return self.get_domains(self.variables)

    @property
    def varsizes(self):
        return self.get_varsizes(self.variables)

    def get_children(self, *variables) -> list:
        return list(dict.fromkeys(sum([list(self.graph.successors(v)) for v in variables], [])))

    def get_parents(self, *variables) -> list:
        return list(dict.fromkeys(sum([list(self.graph.predecessors(v)) for v in variables], [])))

    def markov_blanket(self, v)-> list:
        ch = set(self.get_children(v))
        pa = set(self.get_parents(v))
        pa_ch = set(self.get_parents(*ch)).difference({v})

        # return the union
        return list(ch | pa | pa_ch)

    def get_dag_str(self):
        str_dag = ""
        for v in self.graph.nodes:
            parents = list(self.graph.predecessors(v))
            str_dag += f"[{v}"
            if len(parents) > 0:
                str_dag += "|" + ":".join(parents)
            str_dag += "]"

        return str_dag

    def submodel(self, nodes:list) -> DiscreteDAGModel:
        new_dag = self.graph.subgraph(nodes)
        new_factors = {x: f for x, f in self.factors.items() if x in nodes}
        return self.builder(new_dag, new_factors)

    def rename_vars(self, names_mapping: dict) -> DiscreteDAGModel:
        logging.debug(f"Renaming variables as {names_mapping}")
        new_dag = relabel_nodes(self.graph, names_mapping)
        new_factors = [f.rename_vars(names_mapping) for f in self.factors.values()]
        return self.builder(dag=new_dag, factors=new_factors)


    def sample(self, n_samples: int, as_pandas = False) -> Union[list[Dict], pd.DataFrame]:
        logging.info(f"Sampling {n_samples} instances from model")
        data = forward_sampling(self, n_samples=n_samples)
        if not as_pandas:
            data = data.to_dict("records")
        return data

    def copy(self):
        return self.builder(dag=self.graph, factors=self.factors)









