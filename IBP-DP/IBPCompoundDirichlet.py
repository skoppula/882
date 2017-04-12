import numpy as np
import scipy.special as scis
import csv
# testing
np.seterr(all='raise')

class IBPCompoundDirichlet(object):

	def __init__(self):
		pass
		
	def write_to_file(self, file_root):
		self.get_topics()
		self.get_topic_assignments()

		topics_file = file_root + "_topics"
		assignments_file = file_root + "_assignments"
		pi_phi_file = file_root + "_pi_phi"

		with open(topics_file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter = ' ')
			for i in range(len(self.beta)):
				writer.writerow(self.beta[i])

		with open(assignments_file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter = ' ')
			for i in range(len(self.theta)):
				writer.writerow(self.theta[i])

		with open(pi_phi_file, 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter = ' ')
			writer.writerow(self.pi)
			writer.writerow(self.phi)




	def get_topics(self):
		self.beta = []		
		for k in range(self.n_topics):
			beta_k = []			
			for v in range(self.n_vocab):
				beta_k.append(np.count_nonzero(self.WS[self.ZS == k] == v))
			try:
				beta_k = np.array(beta_k)/sum(beta_k)
			except FloatingPointError:
				beta_k = np.array(beta_k)
			self.beta.append(beta_k)
		self.beta = np.array(self.beta)




	def get_topic_assignments(self):
		self.theta = []
		for m in range(self.n_docs):
			theta_m = []
			for k in range(self.n_topics):
				theta_m.append(np.count_nonzero(self.ZS[self.DS == m] == k))
			try:
				theta_m = np.array(theta_m)/sum(theta_m)
			except FloatingPointError:
				theta_m = np.array(theta_m)
			self.theta.append(theta_m)
		self.theta = np.array(self.theta)


	
	
	def fit_data(self, data, n_iter, gamma, alpha, eta):
		# number of iterations
		self.n_iter = n_iter

		# Set hyperparameters
		self.gamma = gamma
		self.alpha = alpha
		self.eta = eta

		# initialize all latent variables
		self._initialize(data)

		# do posterior inference
		self._posterior_inference()


	def _initialize(self, data):
    
	    # FROM LDA UTIL - for converting word-document matrix into list of words and list of document indices
	    # Rows are documents, columns are vocabulary
	    def _matrix_to_lists(doc_word): 
	        ii, jj = np.nonzero(doc_word)
	        ss = tuple(doc_word[i, j] for i, j in zip(ii, jj))
	        n_tokens = int(doc_word.sum())
	        DS = np.repeat(ii, ss).astype(np.int)
	        WS = np.empty(n_tokens, dtype=np.int)
	        startidx = 0
	        for i, cnt in enumerate(ss):
	            WS[startidx:startidx + cnt] = jj[i]
	            startidx += cnt
	        return WS, DS

	    # Get WS, DS
	    self.WS, self.DS = _matrix_to_lists(data)
	    self.n_words = len(self.WS)
	    print(self.n_words)
	    self.n_docs = self.DS[-1]+1
	    self.n_vocab = max(self.WS)+1

	    self.log = 'perplexity_curve'

	    # Initialize gamma
	    #gamma = np.random.uniform(low = 0.0, high = 50.0)
	    
	    # Simulation for initializing pi
	    def _stick_breaking_sim(min_stick_length=0.01):
	        pi = []
	        stick_length = 1
	        while (stick_length>min_stick_length):
	            mu = np.random.beta(self.alpha,1)
	            pi.append(stick_length*mu)
	            stick_length = stick_length*mu
	        return np.array(pi)
	    
	    # Initialize pi, phi
	    self.pi = _stick_breaking_sim()
	    self.n_topics = len(self.pi)
	    self.phi = np.random.gamma(self.gamma, scale = 1.0, size = (self.n_topics))	# CHANGE TO GAMMA PRIOR
	    
	    # Initialize ZS - list of topic assignments
	    self.ZS = np.random.randint(low = 0, high = self.n_topics, size = self.n_words)


	def _posterior_inference(self):
		for iteration in range(self.n_iter):

			# Gibbs sampling of topic assignments ZS, topic parameters pi and phi

			self._sample_ZS()
			
			print("Iteration", iteration, "Complete z")


			# do not sample new inactive pi, phi if this is the last iteration
			if iteration == self.n_iter-1:
				self._sample_pi_phi(last_iter = True)
			else:
				self._sample_pi_phi()


			print("Iteration", iteration, "Complete pi phi")

			
			perplexity = self.perplexity()
			print("Perplexity:", perplexity)

			with open(self.log, 'a') as file:
				writer = csv.writer(file, delimiter = ' ')
				writer.writerow(["Perplexity:", str(perplexity),"Number of topics:", str(self.n_topics)])
	'''
	def log_likelihood(self):
	    prob_w = []
	    for it in range(10):
	        # sample ZS
	        ZS_sample = self.ZS
	        for i in range(1):
	            ZS_sample = self._sample_ZS_prior(ZS_sample)
	        # compute logp for words given topic assignments and beta
	        logp_w = []
	        for i in range(self.n_words):
	            logp_w.append(np.log(np.count_nonzero(ZS_sample[np.argwhere(self.WS == self.WS[i]).squeeze()] == ZS_sample[i]) + self.eta) - np.log(np.count_nonzero(ZS_sample == ZS_sample[i]) + self.n_vocab*self.eta))
	        prob_w.append(np.exp(np.sum(np.array(logp_w)))

	    print("Likelihood Estimate: ", np.mean(np.array(prob_w)))
	'''

	def log_likelihood(self):
		logp_w = []
		for i in range(self.n_words):
	        	logp_w.append(np.log(np.count_nonzero(self.ZS[self.WS == self.WS[i]] == self.ZS[i]) + self.eta) - np.log(np.count_nonzero(self.ZS == self.ZS[i]) + self.n_vocab*self.eta))
	        	    
		#print("Likelihood Estimate: ", np.sum(np.array(logp_w)))
		return np.sum(np.array(logp_w))

	def perplexity(self):
		return np.exp(-self.log_likelihood()/self.n_words)


	def _sample_ZS_prior(self, ZS):
	    
	    E_theta = np.empty(self.n_topics)

	    # iterate through and update topic assignments

	    for i in range(self.n_words): # i is word index
	        
	        ZS[i] = -1

	        # calculate prob_w, E_theta, paper section 4.1, to approximate PMF for new topic assignment
	        for k in range(self.n_topics):   # k is topic index
	            E_theta[k] = self._compute_E_theta(self.DS[i], k, ZS = ZS)

	        # probability mass function for new topic assignment
	        prob_z_cum = np.cumsum(E_theta/np.sum(E_theta))

	        # choose new topic assignment
	        nran = np.random.rand()
	        ZS[i] = np.argmax(prob_z_cum>nran)
	        
	    return ZS

	def _sample_ZS(self):
    	
	    prob_w = np.empty(self.n_topics)
	    E_theta = np.empty(self.n_topics)
	    
	    # iterate through and update topic assignments

	    for i in range(self.n_words): # i is word index
	        self.ZS[i] = -1
	        if i % 1000 == 0:
	        	print("Sampling", i, "of", self.n_words, "in ZS")
	        """
	        # determine if document index has changed -- if so, need to update E_theta
	        update_E_theta = False
	        if i == 0:
	            update_E_theta = True
	        elif self.DS[i-1] != self.DS[i]:
	            update_E_theta = True
		"""
	        
	        # calculate prob_w, E_theta, paper section 4.1, to approximate PMF for new topic assignment
	        for k in range(self.n_topics):   # k is topic index
	            #prob_w[k] = np.count_nonzero(self.ZS[self.WS == self.WS[i]] == k) + self.eta # ERROR IN PAPER
	            prob_w[k] = np.exp(np.log(np.count_nonzero(self.ZS[self.WS == self.WS[i]] == k) + self.eta) - np.log(np.count_nonzero(self.ZS == k) + self.n_vocab*self.eta))
	            
	            #if update_E_theta:
	            E_theta[k] = self._compute_E_theta(self.DS[i], k, self.ZS)
	        
	        # probability mass function for new topic assignment
	        prob_z = np.multiply(prob_w, E_theta)
	        prob_z_cum = np.cumsum(prob_z/np.sum(prob_z))
	        
	        # choose new topic assignment
	        nran = np.random.rand()
	        self.ZS[i] = np.argmax(prob_z_cum>nran)
	    
	def _compute_E_theta(self, m, k, ZS):
   
	    # 3 cases, 3 separate computations
	    m_idx = [self.DS == m]   # indices for doc m
	    n_m = np.count_nonzero(m_idx) - 1 # number of words in doc m (excluding i)
	    n_km = np.count_nonzero(ZS[m_idx] == k)   # number of words in doc m assigned to topic k
	    n_k = np.count_nonzero(ZS == k)   # number of words in corpus assigned to topic k
	    
	    
	    # Following notation in paper
	    topics_in_m = np.array([i for i in range(self.n_topics) if i in ZS[m_idx]])
	    if len(topics_in_m) == 0:
	        X = 0
	    else:
	    	X = np.sum(self.phi[topics_in_m])
	    
	    # Case 1: kth topic is currently represented in doc m
	    if n_km > 0:
	        # Compute expectation
	        topics_in_Ystar = np.array([i for i in range(self.n_topics) if (i not in ZS[m_idx] and i in ZS)], dtype = np.int)
	        if len(topics_in_Ystar) == 0:
	        	E_ystar = 0
	        else:      
	        	E_ystar = np.sum(np.multiply(self.phi[topics_in_Ystar], self.pi[topics_in_Ystar]))
	        E_ycross = self.alpha * self.gamma
	        E_y = E_ystar + E_ycross
	        
	        # Compute variance
	        if len(topics_in_Ystar) == 0:
	        	V_ystar = 0
	        else:      
	        	V_ystar = np.sum(np.multiply(np.square(self.phi[topics_in_Ystar]), np.multiply(self.pi[topics_in_Ystar], 1 - self.pi[topics_in_Ystar])))
	        V_ycross = self.alpha * self.gamma * (self.gamma + 1) - ((self.alpha ** 2 * self.gamma ** 2) / (2 * self.alpha + 1))
	        V_y = V_ystar + V_ycross
	        
	        # Compute second order Taylor approximation
	        # square brackets in [8]
	        def g(X,Y,n):
	            try:
	            	return 1./(2.**(X+Y)*(n+X+Y))
	            except FloatingPointError:
	            	print (n+X+Y)
	            	exit()

	            
	        # second derivative
	        def d2g(X,Y,n):
	            return 2.**(-X-Y)*(np.log(2.)*(n+X+Y)*(np.log(2.)*(n+X+Y)+2.)+2.)/(n+X+Y)**3.
	        
	        E_theta = (n_km + self.phi[k])*(g(X,E_y,n_m) + d2g(X,E_y,n_m)*V_y/2)
	    
	    # Case 2: kth topic is not in m, but is in corpus
	    elif n_k > 0:
	        # Compute expectation
	        topics_in_Ystar = np.array([i for i in range(self.n_topics) if (i not in ZS[m_idx] and i in ZS and i is not k)], dtype = np.int)
	        topics_in_Ycross = np.array([i for i in range(self.n_topics) if i not in ZS]) # not used
	        if len(topics_in_Ystar) == 0:
	        	E_ystar = 0
	        else:      
	        	E_ystar = np.sum(np.multiply(self.phi[topics_in_Ystar], self.pi[topics_in_Ystar]))
	        E_ycross = self.alpha * self.gamma
	        E_y = E_ystar + E_ycross
	        
	        # Compute variance
	        if len(topics_in_Ystar) == 0:
	        	V_ystar = 0
	        else:      
	        	V_ystar = np.sum(np.multiply(np.square(self.phi[topics_in_Ystar]), np.multiply(self.pi[topics_in_Ystar], 1 - self.pi[topics_in_Ystar])))
	        V_ycross = self.alpha * self.gamma * (self.gamma + 1) - ((self.alpha ** 2 * self.gamma ** 2) / (2 * self.alpha + 1))
	        V_y = V_ystar + V_ycross
	        
	        # Compute second order Taylor approximation
	        def g(X,Y,n):
	            return 1./(2.**(X+Y)*(n+X+Y))
	            
	        # second derivative
	        def d2g(X,Y,n):
	            return 2.**(-X-Y)*(np.log(2.)*(n+X+Y)*(np.log(2.)*(n+X+Y)+2.)+2.)/(n+X+Y)**3.
	        
	        E_theta = self.pi[k]*self.phi[k]*(g(X+self.phi[k],E_y,n_m) + d2g(X+self.phi[k],E_y,n_m)*V_y/2)
	        
	    # Case 3: kth topic not in corpus
	    else:
	        # Compute expectation
	        topics_in_Ystar = np.array([i for i in range(self.n_topics) if (i not in ZS[m_idx] and i in ZS)], dtype = np.int)
	        topics_in_Ycross = np.array([i for i in range(self.n_topics) if i not in ZS]) # not used
	        
	        if len(topics_in_Ystar) == 0:
	        	E_ystar = 0
	        else:      
	        	E_ystar = np.sum(np.multiply(self.phi[topics_in_Ystar], self.pi[topics_in_Ystar]))
	        
	        E_ycross = self.alpha * self.gamma
	        
	        # Compute variance
	        if len(topics_in_Ystar) == 0:
	        	V_ystar = 0
	        else:      
	        	V_ystar = np.sum(np.multiply(np.square(self.phi[topics_in_Ystar]), np.multiply(self.pi[topics_in_Ystar], 1 - self.pi[topics_in_Ystar])))
	        V_ycross = self.alpha * self.gamma * (self.gamma + 1) - ((self.alpha ** 2 * self.gamma ** 2) / (2 * self.alpha + 1))
	        
	        # Compute second order (multivariate) Taylor approximation
	        def g(X,Y,Z,n):
	            return Z/(2.**(X+Y+Z)*(n+X+Y+Z))
	            
	        # second derivative wrt Y
	        def d2gdY2(X,Y,Z,n):
	            return Z*2.**(-X-Y-Z)*(np.log(2.)*(n+X+Y+Z)*(np.log(2.)*(n+X+Y+Z)+2.)+2.)/(n+X+Y+Z)**3.
	        
	        # second derivative wrt Z
	        def d2gdZ2(X,Y,Z,n):
	            return (2.**(-X-Y-Z)*(np.log(2.)*(Z*np.log(2.)-2.)*(n+X+Y+Z)**2.+2.*(Z*np.log(2.)-1.)*(n+X+Y+Z)+2.*Z))/(n+X+Y+Z)**3.
	        
	        E_theta = g(X,E_ystar,E_ycross, n_m) + d2gdY2(X,E_ystar,E_ycross,n_m)*V_ystar/2 + d2gdZ2(X,E_ystar,E_ycross,n_m)*V_ycross/2
	       
	    return E_theta

	def _sample_pi_phi(self, last_iter = False):
    
	    # determine active topics
	    active_topics = np.array([k for k in range(self.n_topics) if np.count_nonzero(self.ZS == k) > 0])
	    
	    # consider phi, pi only for active topics
	    phi = self.phi[active_topics]
	    pi = self.pi[active_topics]
	    
	    # initialize matrix n (position [m,k] is number of words of topic k in document m
	    # reinstantiate B
	    B = np.empty([self.n_docs,len(active_topics)])
	    N = np.empty_like(B)

	    # re-index ZS
	    dict_reindex = dict(zip(active_topics, range(len(active_topics))))
	    self.ZS = np.array([dict_reindex[z] for z in self.ZS])


	    for m in range(B.shape[0]):
	        for k in range(len(active_topics)):
	            N[m,k] = np.count_nonzero(self.ZS[self.DS == m] == k)
	            if N[m,k] > 0:
	                B[m,k] = 1
	            else:
	                nran = np.random.rand()*(pi[k] + 2**phi[k]*(1-pi[k]))
	                if nran > pi[k]:
	                    B[m,k] = 0
	                else:
	                    B[m,k] = 1
	    
	    # sample phi for the active topics (and gamma, delta (TODO, gamma dist rate parameter))
	    phi = self._sample_phi(active_topics, B, N)
	    
	    # sample pi for the active topics
	    pi = self._sample_pi(B)


	    if last_iter:
	    	self.pi = pi
	    	self.phi = phi
	    	self.n_topics = len(self.pi)
	    	return None
	    
	    # generate pi for new topics using slice sampling
	    pi_new = self._generate_new_pi(np.minimum(1,np.amin(pi))) # done using RW metropolis
	    self.pi = np.concatenate([pi, pi_new])

	    # generate phi for new topics
	    phi_new = np.random.gamma(shape = self.gamma, scale = 1., size = pi_new.shape)

	    self.phi = np.concatenate([phi, phi_new])
	    
	    # update number of topics
	    self.n_topics = len(self.pi)
	    
	def _generate_new_pi(self, pi_min): 
	    s = np.random.rand()*pi_min 	# slice variable
	    new_pi = []

	    # Generate new pi according to probability distribution based on semi-ordered stick breaking procedure
	    while pi_min > s:

	    	# Use random walk metropolis hastings to generate each new pi
	        def _metropolis_hastings_rw(pi_min):
	            
	            def f(pi, upper_limit):
	                if 0 < pi < upper_limit:
	                    return np.sum(np.array([(1/i)*(1-pi)**i for i in range(1, self.n_docs+1)])) + (self.alpha-1)*np.log(pi) + self.n_docs*np.log(1-pi)
	                else:
	                    return -np.inf
	                
	            # initialize
	            sample = np.random.rand()*pi_min
	           
	            for it in range(20):
	                new_sample = np.random.rand()*pi_min
	                try:
	                	acceptance_prob = min(1, np.exp(f(new_sample, pi_min)-f(sample, pi_min)))
	                except FloatingPointError:
	                	acceptance_prob = 0

	                if acceptance_prob > np.random.rand():
	                    sample = new_sample
	            return sample
	        
	        pi_min = _metropolis_hastings_rw(pi_min)
	        new_pi.append(pi_min)
	        
	    return np.array(new_pi[0:-1])

	def _sample_pi(self, B):
		pi = np.empty(B.shape[1])
		for k in range(len(pi)):
	            pi[k] = np.random.beta(np.sum(B[:,k]), 1 + self.n_docs - np.sum(B[:,k]))
		return pi

	def _sample_phi(self, active_topics, B, N):
		phi = []

		def _metropolis_hastings_rw(B, N, k, active_topics):

			def f(phi, B, N):
				if phi <= 0:
					return -np.inf

				logp_phi_given_gamma = np.sum((self.gamma-1)*np.log(phi)-phi)
				logp_n_given_phiB = np.sum(np.multiply(B,np.log(scis.gamma(phi+N))-np.log(scis.gamma(phi))-np.log(2)*phi))

				return logp_phi_given_gamma + logp_n_given_phiB

			# initialize
			sample = self.phi[active_topics[k]]

			for it in range(100):
				new_sample = np.random.normal()*2+sample
				try:
					acceptance_prob = min(1, np.exp(f(new_sample, B[:,k], N[:,k])-f(sample, B[:,k], N[:,k])))
				except FloatingPointError:
					acceptance_prob = 0
				if acceptance_prob > np.random.rand():
					sample = new_sample
			return sample


		for k in range(len(active_topics)):
			phi.append(_metropolis_hastings_rw(B,N,k, active_topics))

		return np.array(phi)



