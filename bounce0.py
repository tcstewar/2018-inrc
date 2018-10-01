import nengo
import numpy as np

np.random.seed(1)
class Environment(object):
    x = 0
    y = 0
    vx = np.random.uniform(-1, 1)
    vy = np.random.uniform(-1, 1)
    history = np.zeros((1000, 4))

    def update(self, t, predict):
        dt = 0.001
        self.vx += np.random.normal(0, 0.01)
        self.vy += np.random.normal(0, 0.01)

        self.x += self.vx * dt
        self.y += self.vy * dt
        while self.x > 1:
            self.x = 1 - (self.x-1)
            self.vx = -self.vx
        while self.x < -1:
            self.x = -1 - (self.x+1)
            self.vx = -self.vx
        while self.y > 1:
            self.y = 1 - (self.y-1)
            self.vy = -self.vy
        while self.y < -1:
            self.y = -1 - (self.y+1)
            self.vy = -self.vy

        self.history[1:] = self.history[:-1]
        self.history[0] = self.x, self.y, self.vx, self.vy

        path = []
        for i in range(0, 1000, 100):
            path.append('<circle cx={} cy={} r=1 style="fill:black"/>'.format(self.history[i,0]*100,self.history[i,1]*100))
        for i in range(10):
            path.append('<circle cx={} cy={} r=1 style="fill:yellow"/>'.format(predict[2*i]*100,predict[2*i+1]*100))

        Environment.update._nengo_html_ = '''
        <svg width=100% height=100% viewbox="-100 -100 200 200">
            <rect x=-100 y=-100 width=200 height=200 style="fill:green"/>
            <circle cx={} cy={} r=5 style="fill:white"/>
            {}           
        </svg>
        '''.format(self.history[-1,0]*100, self.history[-1,1]*100, ''.join(path))
            
        future = []
        for i in range(0, 1000, 100):
            future.extend(self.history[i, :2])
        return np.hstack([self.history[-1], future])


model = nengo.Network()
with model:
    env = Environment()
    for i in range(1000):
        env.update(0, np.zeros(20))
        
    env_node = nengo.Node(env.update, size_in=20)
    
    ens = nengo.Ensemble(n_neurons=10, dimensions=1)
    
