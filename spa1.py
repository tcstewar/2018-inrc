import nengo_spa as spa

model = spa.Network()
with model:
    D = 64
    color = spa.State(D, label='color')
    shape = spa.State(D, label='shape')
    memory = spa.State(D, label='memory', feedback=1)
    
    color*shape >> memory
    
    question = spa.State(D, label='question')
    answer = spa.State(D, label='answer')
    
    memory*~question >> answer

print('number of neurons:', model.n_neurons)