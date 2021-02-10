import numpy as np

class LinUCB():
    
    def __init__(self, z_dim, action_dim, delta=0.05):
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.alpha = 1.0 + np.sqrt(np.log(2 / delta) / 2)
        self.reset()
    
    def reset(self):
        self.A = [np.eye(self.z_dim) for i in range(self.action_dim)]
        self.b = [np.zeros(self.z_dim) for i in range(self.action_dim)]
    
    def select_action(self, z_t):
        theta = np.zeros((self.action_dim, self.z_dim))
        ucb = np.zeros(self.action_dim)
        for a in range(self.action_dim):
            A_inv = np.linalg.inv(self.A[a])
            theta[a] = np.dot(A_inv, self.b[a])
            ucb[a] = np.dot(z_t[a], theta[a]) + self.alpha * np.sqrt(np.dot(np.dot(z_t[a], A_inv), z_t[a]))
        
        return np.argmax(ucb)
    
    def update(self, data):
        _, z_t_a_t, a_t, r_t = data
        assert z_t_a_t.shape == (self.z_dim, )
        assert type(a_t) == np.int64
        assert type(r_t) == float
        self.A[a_t] += np.dot(z_t_a_t.reshape((self.z_dim, 1)), z_t_a_t.reshape((1, self.z_dim)))
        self.b[a_t] += r_t * z_t_a_t
    
    
class TS():
    
    def __init__(self, z_dim, action_dim, beta=0.0):
        self.a = None
        self.b = None
        self.mu = None
        self.sigma2 = None
        

    
    def reset(self):
        pass
    
    def select_action(self, z_t):
        pass
    
    def update(self, data):
        pass
