import nengo_spa as spa

model = spa.Network()
with model:
    D = 32
    vision = spa.State(D, label='vision')
    memory = spa.State(D, feedback=1, label='memory')
    vision*0.1 >> memory
    
print('number of neurons:', model.n_neurons)