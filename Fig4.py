import agentpy as ap
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class Cell(ap.Agent):

    def setup(self):
        """ Initialization """
        self.condition = 0  # Normal = 0, Cancer = 1
        self.trait = 0  # Measure of corruption: 0 to 1

    def interact_mut(self,cancer_frac,trait_mean):
        """ Mutation or interaction with other cells """
        rng = self.model.random
        mutation_rate = self.p.mutation_rate
        if mutation_rate > rng.random(): # Mutation
            if self.condition == 0:
                self.condition = 1
            else:
                self.trait = min(self.trait+self.p.mut_size,1)
        else:
            nbors = self.network.neighbors(self)
            for n in nbors:
                if self.condition==n.condition==0: # Normal-Normal
                    pass
                elif self.condition==n.condition==1: # Cancer-Cancer
                    if self.p.cancer_interact > rng.random():
                        if self.trait<n.trait:
                            self.trait+=(n.trait-self.trait)/2
                        else:
                            n.trait+=(self.trait-n.trait)/2
                    else:
                        pass
                elif self.condition == 0 and n.condition == 1: #self normal, other cancer
                    cn_interact = np.exp(-2*n.trait)-0.1 # Probability of normal-cancer interaction
                    kill_prob = 1.05-np.exp(-1.2*n.trait) # Probability of death
                    conv_prob = .2*np.tanh(n.trait) # Cancer exosome effect
                    if cn_interact > rng.random():
                        a = rng.random()
                        if kill_prob > a:
                            n.condition = 0
                            n.trait = 0
                        elif conv_prob+kill_prob > a:
                            self.condition = 1
                        else: # Reversion
                            n.trait/=2 # Make less extreme
                    else:
                        pass
                elif self.condition == 1 and n.condition == 0: #self cancer, other normal
                    cn_interact = np.exp(-2*self.trait)-0.1 # Probability of cancer-normal interaction
                    kill_prob = 1.05 - np.exp(-1.2 * self.trait)  # Probability of death
                    conv_prob = .2 * np.tanh(self.trait)  # Cancer exosome effect
                    if cn_interact > rng.random():
                        a = rng.random()
                        if kill_prob > a:
                            self.condition = 0
                            self.trait = 0
                        elif conv_prob+kill_prob > a:
                            n.condition = 1
                        else:
                            self.trait/=2 # Make less extreme
                    else:
                        pass

class CancerModel(ap.Model):

    def setup(self):
        """ Initialize the agents and network of the model. """

        # Prepare a network
        graph = nx.gnm_random_graph(self.p.population,self.p.population*self.p.average_node_degree/2)

        # Create agents and network
        self.agents = ap.AgentList(self, self.p.population, Cell)
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

        # Initialize cancer in a random share of the population
        C0 = int(self.p.initial_cancer_share * self.p.population)
        for i in self.agents.random(C0):
            i.condition = 1
            i.trait = self.model.random.random() # Generating random initial corruption levels

    def update(self):
        """ Record variables after setup and each step. """
        # Record share of agents in normal and cancer states
        for i, c in enumerate(('N', 'C')):
            n_agents = len(self.agents.select(self.agents.condition == i))
            self[c] = n_agents / self.p.population
            self.record(c)

        # Stop simulation if cancer or normal cells take over
        if self.N == 0 or self.C == 0:
            self.stop()

    def step(self):
        """ Define the models' events per simulation step. """
        tot_canc = len(self.agents.select(self.agents.condition == 1))
        cancer_frac = tot_canc/self.p.population
        trait_mean = 0
        for i in self.agents.select(self.agents.condition == 1):
            trait_mean+=i.trait
        trait_mean = trait_mean/tot_canc

        self.agents.interact_mut(cancer_frac,trait_mean)

    def end(self):
        """ Record evaluation measures at the end of the simulation. """
        self.report('Total share cancerous', self.C)

parameters = {
    'population': 1000,
    'initial_cancer_share':.1,
    'cancer_interact': .9,
    'mut_size': .3,
    'mutation_rate':ap.Range(0,.2),
    'steps':1000,
    'average_node_degree':ap.Range(0,8)
}

sample = ap.Sample(
    parameters,
    n=64, #128
    method='saltelli',
    calc_second_order=False
)

exp = ap.Experiment(CancerModel, sample, iterations=10)
results = exp.run()
results.save()
results = ap.DataDict.load('CancerModel')
results.reporters.hist()
plt.show()

def plot_sensitivity(results):
    """ Show average simulation results for different parameter values. """

    sns.set()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    data = results.arrange_reporters().astype('float')
    params = results.parameters.sample.keys()

    for x, ax in zip(params, axs):
        for y in results.reporters.columns:
            sns.regplot(x=x, y=y, data=data, ax=ax, ci=95,x_bins=15, fit_reg=False, label=y)
        ax.set_ylim(0,1.1)
        ax.set_ylabel('Total Share Cancerous')

    plt.tight_layout()

plot_sensitivity(results)
plt.show()
