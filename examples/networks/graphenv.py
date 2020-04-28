import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

ADD_EDGE_PROBABILITY = 0.25
SCALE_REMOVE_EDGE_PROBABILITY = 1.00
REMOVE_EDGE_PROBABILITY = 0.1


class GraphBestofNEnvironment:
    def __init__(self,num_agents, num_sites, attach_mode, detach_mode, seed=123):
        """Create a bi-partite graph with num_agents and num_sites. This version of the file
        uses a factory design pattern for selecting agents and sites to attach, and for selecting
        agents and sites to detach. Contrast to ConfigurationGraph.py, which doesn't
        use the factory design pattern."""
        if seed is None:
            self.nprandom = np.random.RandomState()   # pylint: disable=E1101
        else:
            self.nprandom = np.random.RandomState(    # pylint: disable=E1101
                seed)
        self.seed = seed
        self.NumAgents = num_agents
        self.state = np.zeros(num_agents, dtype=np.int16)
        self.NumSites = num_sites
        self.color_map = []

        # Interfaces for factory design patterh for attaching and detaching.
        # To attach, call attach_Agent. To detach, call detach_agent.
        self._testIfAttachModeIsLegal(attach_mode) # Will exit if not a legal attachment mode according to factory design pattern
        self._testIfDetachModeIsLegal(detach_mode) # Will exit if not a legal detachment mode according to factory design pattern
        self.attachAgent = self._getAttachmentInterface(attach_mode)
        self.detachAgent = self._getDetachmentInterface(detach_mode)
        # agent vertices
        self.AgentList = []
        for i in range(0,self.NumAgents):
            self.AgentList.append('a'+str(i))
            #self.color_map.append('blue')
            self.color_map.append((0,179/255,179/255))

        #site vertices
        self.SiteList = []
        for i in range(0, self.NumSites):
            self.SiteList.append('s'+str(i))
            #self.color_map.append('magenta')
            self.color_map.append((179/255,i/self.NumSites,10/255))

        # Graph elements
        self.G = nx.Graph()
        self.G.add_nodes_from(self.AgentList, bipartite=0)
        self.G.add_nodes_from(self.SiteList, bipartite=1)

    def addEdge(self,agentVertex,siteVertex):
        agentLabel = 'a'+str(agentVertex)
        siteLabel = 's'+str(siteVertex)
        self.G.add_edge(agentLabel,siteLabel, weight = self.nprandom.randn(1,1))

    """ Utilities used by the Factory Design Pattern for Detachment"""
    def _getDetachmentInterface(self,detachment_type):
        """ return the method that is to be used to perform the detachment process """
        if detachment_type == 'uniform':
            return self._detachAgentUniform
        elif detachment_type == 'linear':
            return self._detachAgentLinear
        elif detachment_type == 'exponential':
            return self._detachAgentExponential
        elif detachment_type == 'power law':
            return self._detachAgentPowerLaw
        elif detachment_type == 'perfect':
            return self._detachAgentPerfect
        #else:
        print('choices are \'uniform\', \'linear\', \'exponential\', \'power law\', or \'perfect\'' )
        return False

    def _testIfDetachModeIsLegal(self,detach_mode):
        if detach_mode == 'linear' or detach_mode == 'uniform' or detach_mode == 'exponential' or detach_mode == 'power law' or detach_mode == 'perfect':
            return True
        else:
            print('Illegal detachment mode ', detach_mode)
            print('Mode must be \'uniform\', \'linear\', \'exponential\', \'importance linear\', or \'importance exponential\'')
            exit

    def _getRandomEdge(self):
        edges = [e for e in self.G.edges]
        if len(edges) <=0: return -1, -1
        edge_to_remove = edges[self.nprandom.randint(len(edges))]
        site_to_remove = edge_to_remove[1] # sites have names 'si' where 's' is the character s and 'i' is an integer
        site_quality = int(site_to_remove[1:len(site_to_remove)]) # Take off the leading 's' and make the number an int
        # Recall that the site-quality is just given by the site number, with higher number indicating higher quality
        return edge_to_remove, site_quality

    """ Collection of Interfaces for Factory Design Pattern for Detachment """
    def _detachAgentUniform(self, edge_to_remove, site_quality):
        """ Remove an edge with a certain probability regardless of quality of the site """
        # edge_to_remove, site_quality = self._getRandomEdge()
        # if edge_to_remove == -1: return # No edges in graph, so just return
        if self.nprandom.uniform(0,1) <= REMOVE_EDGE_PROBABILITY:
            self.G.remove_edges_from([edge_to_remove])

    def _detachAgentLinear(self, edge_to_remove, site_quality):
        """ Remove an edge with a linearly decreasing probability of site quality.
        The equation is p_{detach}(a,s) = qual(s)/NumSites. Recall that qual(s)
        is simply the index of the site, ranging from 0 to NumSites-1. This means
        that the lowest quality site is always removed. Could consider a scaling
        factor. When there are a lot of sites, many high quality
        sites will rarely be removed."""
        # edge_to_remove, site_quality = self._getRandomEdge()
        # if edge_to_remove == -1: return # No edges in graph, so just return
        if self.nprandom.uniform(0,1)>= site_quality/self.NumSites:
            self.G.remove_edges_from([edge_to_remove])

    def _detachAgentExponential(self, edge_to_remove, site_quality):
        """ Remove an edge with exponentially decreasing probability of site quality.
        The equation is p_{detach}(a,s) = exp(-4*qual(site)/(NumSites-1). Recall that qual(s)
        is simply the index of the site, ranging from 0 to NumSites-1. This means
        that the highest quality site is removed with probability exp(-4) and
        the lowest quality site is removed with probability 1.
        Could consider a scaling factor. When there are a lot of sites, many high quality
        sites will rarely be removed."""
        # edge_to_remove, site_quality = self._getRandomEdge()
        # if edge_to_remove == -1: return # No edges in graph, so just return
        magic_number = -4 # TODO: remove this magic number from the exponential decrease in attachment
        threshold = 1.0-np.exp(magic_number*site_quality/(self.NumSites-1))
        if self.nprandom.uniform(0,1) >= threshold:
            self.G.remove_edges_from([edge_to_remove])
        print('In _detachAgentExponential method. To be completed')

    def _detachAgentPowerLaw(self, edge_to_remove, site_quality):
        """ Remove an edge with power law decreasing probability of site quality.
        The equation is p_{detach}(a,s) = 1-2^{-3*qual(site)/NumSites}. Recall that qual(s)
        is simply the index of the site, ranging from 0 to NumSites-1. This means
        that the highest quality site is removed with a small probability and
        the lowest quality site is removed with probability 1.
        Unlike the other detachment probabilities, this appears to scale well with
        the number of sites."""
        # edge_to_remove, site_quality = self._getRandomEdge()
        # if edge_to_remove == -1: return # No edges in graph, so just return
        magic_number_basis = 2 # TODO: remove magic number from power law equation
        magic_number_scale = -5 # TODO: remove magic number from power law equation
        if self.nprandom.uniform(0,1)>= 1.0 - magic_number_basis**(magic_number_scale*site_quality/self.NumSites):
            self.G.remove_edges_from([edge_to_remove])

    def _detachAgentPerfect(self, edge_to_remove, site_quality):
        """ Remove an edge with a linearly decreasing probability of site quality.
        The equation is p_{detach}(a,s) = (1+qual(s))/NumSites. Recall that qual(s)
        is simply the index of the site, ranging from 0 to NumSites-1. This means
        that the highest quality site is **never*** removed. Could consider a scaling
        factor. When there are a lot of sites, many high quality
        sites will rarely be removed."""
        # edge_to_remove, site_quality = self._getRandomEdge()
        # if edge_to_remove == -1: return # No edges in graph, so just return
        if self.nprandom.uniform(0,1) >= (1+site_quality)/self.NumSites:
            self.G.remove_edges_from([edge_to_remove])

    """ Utilities used by the Factory Design Pattern for Attachment"""
    def _testIfAttachModeIsLegal(self,attach_mode):
        if attach_mode == 'linear' or attach_mode == 'always' or attach_mode == 'exponential' or attach_mode == 'importance linear' or attach_mode == 'importance exponential' or attach_mode == 'importance 2 linear' or attach_mode == 'importance 2 exponential':
            return True
        else:
            print('Illegal attachment mode ', attach_mode)
            print('Mode must be \'always\', \'linear\', \'exponential\', \'importance linear\',\'importance 2 linear\', \'importance exponential\', or \'importance 2 exponential\'')
            exit

    def _getRandomAgent(self):
        """ Interface for randomly selecting an agent node from the graph.
        The algorithm selects one of the NumAgents of agents, appends an 'a' prefix,
        and returns the corresponding labeled agent, like 'a2'"""
        random_agent = 'a' + str(self.nprandom.randint(self.NumAgents))
        if self.G.degree(random_agent) > 0:
            # If agent is already attached, don't return a valid agent
            return None
        else: return random_agent

    def _getRandomSite(self):
        """ Interface for randomly selecting site node from the graph.
        The algorithm selects one of the NumSites of sitess, appends an 's' prefix,
        and returns the corresponding labeled site, like 's1'"""
        #self.Edge2DetachInterface = self._edgeDetachRandom
        return 's' + str(self.nprandom.randint(self.NumSites))

    def _findSiteFromDegreePDF(self, pdf, D, site_degree_list):
        cdf=[sum(pdf[0:i+1]) for i in range(0,len(pdf))] # create cumulative distribution function (cdf) from pmf

        #***** STEP 4: Sample a degree with probability proportional to f *****#
        r = self.nprandom.uniform(0,1) # uniform random variable from interval [0,1]
        myMin = 1.1
        myIndex = self.NumSites + 1
        for i in range(len(cdf)): # find the minimum value of the pmf that exceeds the threshold
            Delta = cdf[i] - r
            if Delta >=0 and Delta < myMin:
                myMin = Delta
                myIndex = i
                #break # exit for loop since first degree found is the lowest degree and should be selected.
        chosen_d = D[myIndex]
        #***** STEP 5: find all the sites that have the degree chosen in STEP 4 *****#
        matchingSiteIndices = []
        for i in range(len(site_degree_list)):
            if site_degree_list[i] == chosen_d:
                matchingSiteIndices.append(i)  # Add the index of sites that have a matching degree
        #***** STEP 6: select one of the sites with matching degree and attach to it *****#
        chosenSiteIndex = self.nprandom.randint(matchingSiteIndices[0],matchingSiteIndices[-1]+1)
        site = 's' + str(chosenSiteIndex)
        return site

    def _findSortedDegreeList(self):
        degree_list = [d for n, d in self.G.degree()]  # degree sequence
        site_degree_list = degree_list[self.NumAgents:self.NumAgents+self.NumSites] # Cull out sites
        D = sorted(list(set(site_degree_list))) # Sorted list of degree values
        return D, site_degree_list

    def _getAttachmentInterface(self, attachment_type):
        """ return the method that is to be used to perform the attachment process """
        if attachment_type == 'always':
            return self._attachAgentAlways
        elif attachment_type == 'linear':
            return self._attachAgentLinear
        elif attachment_type == 'exponential':
            return self._attachAgentExponential
        elif attachment_type == 'importance linear':
            return self._attachAgentImportanceLinear
        elif attachment_type == 'importance exponential':
            return self._attachAgentImportanceExponential
        elif attachment_type == 'importance 2 linear':
            return self._attachAgentImportanceLinear2Groups
        elif attachment_type == 'importance 2 exponential':
            return self._attachAgentImportanceExponential2Groups
        #else:
        print('choices are \'always\', \'linear\', \'exponential\', \'importance linear\', \'importance exponential\', or \'importance 2 exponential\'' )
        return False

    """ Collection of Interfaces for Factory Design Pattern for Attachment """
    def _attachAgentAlways(self, random_agent):
        """There are four attachment types: always, linear, exponential, and importance
        sampling. This method always attaches an agent to a random site unless the agent
        is already attached to a site. Part of factory design pattern for attaching."""
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #    return
        random_site = self._getRandomSite()
        random_weight = 1
        self.G.add_edge(random_agent,random_site,weight=random_weight)

    def _attachAgentLinear(self, random_agent):
        """There are fpur attachment types: always, linear, exponential, and importance
        sampling. This method attaches an agent to a random site with probability linearly
        proportional to the degree of the site unless the agent is already attached to a site.
        Part of factory design pattern for attaching.

        The equation is p(agent,site) = alpha_{attach} [1+deg(site)]."""
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #     return
        random_site = self._getRandomSite()
        random_weight = 1
        if self.nprandom.uniform(0,1)<= ADD_EDGE_PROBABILITY*(1+self.G.degree(random_site)):
            self.G.add_edge(random_agent,random_site,weight=random_weight)

    def _attachAgentExponential(self, random_agent):
        """There are four attachment types: always, linear, exponential, and importance
        sampling. This method attaches an agent to a random site with probability proportional
        to an exponential function of the degree of the site, unless the agent is already
        attached to a site. Part of factory design pattern for attaching.

        The equation is p(agent,site) = max{alpha,exp[-beta(N-1-deg(site))/N]}, where beta is a
        magic number, N is the number of agents, and alpha is a minimum probability of attaching. """
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #    return
        random_site = self._getRandomSite()
        random_weight = 1
        magic_number = 3 # TODO: replace this magic_number with an adjustable parameter
        threshold = np.exp(-1.0* magic_number*(self.NumAgents-1-self.G.degree(random_site))/self.NumAgents)
        if self.nprandom.uniform(0,1) <= max(ADD_EDGE_PROBABILITY,threshold):
            self.G.add_edge(random_agent,random_site,weight=random_weight)

    def _attachAgentImportanceLinear(self, random_agent):
        """There are four attachment types: always, linear, exponential, and importance
        sampling. This method attaches an agent to a site using importance smapling,
        unless the agent is already attached to a site. The importance sampling means that
        an agent will always attach to a site (unless the agent is already attached), and the
        probability of attaching to a particular site is proportional to the degree of that site.
        There are two forms of importance sampling, one linearly proportional to site degree
        and the other exponentially proportional to site degree.

        Part of factory design pattern for attaching.

        The algorithm steps are annotated by comments """
        #***** STEP 1: Choose a random agent *****#
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #     return
        random_weight = 1
        #***** STEP 2: Create and sort degree set *****#
        D, siteDegreeList  = self._findSortedDegreeList()    # Sorted list of degree values
        #***** STEP 3: Create monotonically increasing probability of degree *****#
        pdf = [d + 1/len(D) for d in D]   # f(d) = d + 1/|D|, linear
        pdf = [value /sum(pdf) for value in pdf] # normalize to create probability mass function (pmf)
        #***** STEP 4-6: Implemented in _findSiteDromDegreePDF method *****#
        site = self._findSiteFromDegreePDF(pdf,D,siteDegreeList)
        self.G.add_edge(random_agent,site,weight=random_weight)

    def _attachAgentImportanceExponential(self, random_agent):
        """There are four attachment types: always, linear, exponential, and importance
        sampling. This method attaches an agent to a site using importance smapling,
        unless the agent is already attached to a site. The importance sampling means that
        an agent will always attach to a site (unless the agent is already attached), and the
        probability of attaching to a particular site is proportional to the degree of that site.
        There are two forms of importance sampling, one linearly proportional to site degree
        and the other exponentially proportional to site degree.

        Part of factory design pattern for attaching.

        The algorithm steps are annotated by comments """
        #***** STEP 1: Choose a random agent *****#
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #    return
        random_weight = 1
        #***** STEP 2: Create and sort degree set *****#
        D, siteDegreeList  = self._findSortedDegreeList()    # Sorted list of degree values
        #***** STEP 3: Create monotonically increasing probability of degree *****#
        pdf = [np.exp(d) for d in D]   # f(d) = exp(d), exponential
        pdf = [value /sum(pdf) for value in pdf] # normalize to create probability mass function (pmf)
        #***** STEP 4-6: Implemented in _findSiteDromDegreePDF method *****#
        site = self._findSiteFromDegreePDF(pdf, D, siteDegreeList)
        self.G.add_edge(random_agent,site,weight=random_weight)

    def _attachAgentImportanceLinear2Groups(self, random_agent):
        """This method attaches an agent to a site using importance smapling,
        unless the agent is already attached to a site. The importance sampling means that
        an agent will always attach to a site (unless the agent is already attached), and the
        probability of attaching to a particular site is proportional to the degree of that site.
        There are two forms of importance sampling, one linearly proportional to site degree
        and the other exponentially proportional to site degree.

        The equation is min(exp(deg),exp(NumAgents-deg)), which increases from zero until
        NumAgents/2, and then decreases in a symmetric way from NumAgents/2 to NumAgents.

        Part of factory design pattern for attaching.

        The algorithm steps are annotated by comments """
        #***** STEP 1: Choose a random agent *****#
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #    return
        random_weight = 1
        #***** STEP 2: Create and sort degree set *****#
        D, siteDegreeList  = self._findSortedDegreeList()    # Sorted list of degree values
        #***** STEP 3: Create monotonically increasing probability of degree *****#
        pdf = [d + 1/len(D) for d in D]   # f(d) = exp(d), exponential
        for i in range(len(pdf)):  # zero out high degrees in pdf so that there is no probability of getting too high of a degree
            if D[i] >self.NumAgents/2.0:
                pdf[i]=0
        pdf = [value /sum(pdf) for value in pdf] # normalize to create probability mass function (pmf)
        #***** STEP 4-6: Implemented in _findSiteDromDegreePDF method *****#
        site = self._findSiteFromDegreePDF(pdf, D, siteDegreeList)
        self.G.add_edge(random_agent,site,weight=random_weight)

    def _attachAgentImportanceExponential2Groups(self, random_agent):
        """This method attaches an agent to a site using importance smapling,
        unless the agent is already attached to a site. The importance sampling means that
        an agent will always attach to a site (unless the agent is already attached), and the
        probability of attaching to a particular site is proportional to the degree of that site.
        There are two forms of importance sampling, one linearly proportional to site degree
        and the other exponentially proportional to site degree.

        The equation is min(exp(deg),exp(NumAgents-deg)), which increases from zero until
        NumAgents/2, and then decreases in a symmetric way from NumAgents/2 to NumAgents.

        Part of factory design pattern for attaching.

        The algorithm steps are annotated by comments """
        #***** STEP 1: Choose a random agent *****#
        # random_agent = self._getRandomAgent()
        # if random_agent == None:
        #    return
        random_weight = 1
        #***** STEP 2: Create and sort degree set *****#
        D, siteDegreeList  = self._findSortedDegreeList()    # Sorted list of degree values
        #***** STEP 3: Create monotonically increasing probability of degree *****#
        pdf = [np.exp(d) for d in D]   # f(d) = exp(d), exponential
        for i in range(len(pdf)):  # zero out high degrees in pdf so that there is no probability of getting too high of a degree
            if D[i] >self.NumAgents/2.0:
                pdf[i]=0
        pdf = [value /sum(pdf) for value in pdf] # normalize to create probability mass function (pmf)
        #***** STEP 4-6: Implemented in _findSiteDromDegreePDF method *****#
        site = self._findSiteFromDegreePDF(pdf, D, siteDegreeList)
        self.G.add_edge(random_agent,site,weight=random_weight)

    def resetGraph(self):
        edges = [e for e in self.G.edges]
        self.G.remove_edges_from(edges)

    def reset(self):
        self.resetGraph()

    def showGraph(self):
        #pos = dict()
        #pos.update( (n, (1, i)) for i, n in enumerate(self.AgentList) ) # put nodes from X at x=1
        #pos.update( (n, (2, i)) for i, n in enumerate(self.SiteList) ) # put nodes from Y at x=2
        #nx.draw(self.G, pos=pos, with_labels=True)
        #nx.draw_spring(self.G, with_labels=True)
        nx.draw_circular(self.G, with_labels=True,node_color = self.color_map,weight='weight')
        plt.pause(.001)
        #plt.show()
        plt.clf()

    def showGraphFinal(self):
        #pos = dict()
        #pos.update( (n, (1, i)) for i, n in enumerate(self.AgentList) ) # put nodes from X at x=1
        #pos.update( (n, (2, i)) for i, n in enumerate(self.SiteList) ) # put nodes from Y at x=2
        #nx.draw(self.G, pos=pos, with_labels=True)
        #nx.draw_spring(self.G, with_labels=True)
        nx.draw_circular(self.G, with_labels=True,node_color = self.color_map,weight='weight')
        # plt.pause(.001)
        plt.show()
        # plt.clf()

    def toString(self):
        degree_list = self.getVertexDegree()
        site_list = degree_list[self.NumAgents:self.NumAgents+self.NumSites]
        #print("Site degree list",site_list)
        for s,n in site_list: print(s,", ",n/self.NumAgents)

    def isGraphSuccessful(self):
        """The success criterion is whether the highest quality site has the highest degree.
        The isGraphSuccessful method returns 0 if unsuccessful and 1 if successful. Success
        is true even if the highest quality site is tied with other sites for the highest degree."""
        site_list = self.getSiteDegree()
        highest_degree = max([degree for name, degree in site_list])
        #print("Highest degree is ",highest_degree,"in Degree list = ",site_list)
        degreeTopSite = self.getTopSiteDegree()
        #print("Degree of highest quality site is",degreeTopSite)
        if degreeTopSite == highest_degree:
            return 1
        else:
            return 0

    def getNumAgents(self):
        return self.NumAgents

    def getNumSites(self):
        return self.NumSites

    def getVertexDegree(self):
        """Returns a list of vertices and their degrees"""
        L = list(self.G.degree())
        #print("degree list", L)
        return L

    def getIncidenceMatrix(self):
        M = list(nx.incidence_matrix(self.G,'weight'))
        print("incidence matrix:\n",M)

    def getAdjacencyMatrix(self):
        M = nx.to_numpy_matrix(self.G,weight='weight')
        print("adjacency matrix:\n",M)
        print("sliced adjacency matrix:\n",M[0:self.NumAgents,self.NumAgents:self.NumAgents+self.NumSites])

    def getSiteDegree(self):
        degree_list = self.getVertexDegree()
        site_list = degree_list[self.NumAgents:self.NumAgents+self.NumSites]
        return site_list

    def getTopSiteDegree(self):
        degree_list = self.getVertexDegree()
        topSiteDegree = degree_list[-1]
        return topSiteDegree[1]

    def setStateSpace(self):
        self.state = np.zeros(self.NumAgents, dtype=np.int16)
        for agent, site in self.G.edges:
            self.state[int(agent[1:])] = int(site[1:])+1

    def step(self):
        # Get a random agent
        random_agent = 'a' + str(self.nprandom.randint(self.NumAgents))

        ## Check if the agent is attached to a site or not
        # If the agent is attached to a site do the dis-attachment process
        if self.G.degree(random_agent) == 0:
            self.attachAgent(random_agent)
        # Else do the deattachment process
        else:
            edge_to_remove = [val for val in self.G.edges if val[0]==random_agent][0]
            site_to_remove = edge_to_remove[1] # sites have names 'si' where 's' is the character s and 'i' is an integer
            site_quality = int(site_to_remove[1:len(site_to_remove)]) # Take off the leading 's' and make the number an int
            self.detachAgent(edge_to_remove, np.power(site_quality, site_quality))
        self.setStateSpace()
        return self.state, 0.0, False, dict()
