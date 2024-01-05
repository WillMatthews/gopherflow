package nn

import (
    "fmt"
    "math/rand"
    "github.com/WillMatthews/gopherflow/internal/engine"
)

type Neuron struct {
    weight float64
    bias float64
    activation string
}

func (n *Neuron) newNeuron( activation string, N int) *Neuron {
    weights := make([]Value, N)
    for i := 0; i < N; i++ {
        weights[i] = Value{data: rand.Float64()*2-1}
    }

    return &Neuron{
        weight: weights,
        bias: Value{data: 0},
        activation: activation,
    }
}

func (n *Neuron) run(inputs []float64) float64 {
    sum := 0.0
    for i := 0; i < len(inputs); i++ {
        sum += inputs[i] * n.weight[i]
    }

    result := sum + n.bias

    switch n.activation {
    case "relu":
        return Relu(result)
    case "sigmoid":
        return Sigmoid(result)
    case "tanh":
        return Tanh(result)
    default:
        return result
}

func (n *Neuron) params() []Value {
    return append(n.weight, n.bias)
}

type Layer struct {
    neurons []Neuron
}

func (l *Layer) newLayer(activation string, Ni, No int) *Layer {
    neurons := make([]Neuron, No)
    for i := 0; i < No; i++ {
        neurons[i] = Neuron.newNeuron(activation, Ni)
    }

    return &Layer{
        neurons: neurons,
    }
}

func (l *Layer) run(inputs []float64) []float64 {
    outputs := make([]float64, len(l.neurons))
    for i := 0; i < len(l.neurons); i++ {
        outputs[i] = l.neurons[i].run(inputs)
    }

    return outputs
}

func (l *Layer) params() []Value {
    params := make([]Value, 0)
    for i := 0; i < len(l.neurons); i++ {
        params = append(params, l.neurons[i].params()...)
    }

    return params
}

type MLP struct {
    layers []Layer
}

func (m *MLP) newMLP(activation string, Ni, Nh, No int) *MLP {
    layers := make([]Layer, 2)
    layers[0] = Layer.newLayer(activation, Ni, Nh)
    layers[1] = Layer.newLayer(activation, Nh, No)

    return &MLP{
        layers: layers,
    }
}

func (m *MLP) run(inputs []float64) []float64 {
    outputs := inputs
    for i := 0; i < len(m.layers); i++ {
        outputs = m.layers[i].run(outputs)
    }

    return outputs
}

func (m *MLP) params() []Value {
    params := make([]Value, 0)
    for i := 0; i < len(m.layers); i++ {
        params = append(params, m.layers[i].params()...)
    }

    return params
}

func (m *MLP) loss(inputs []float64, targets []float64) float64 {
    outputs := m.run(inputs)
    loss := 0.0
    for i := 0; i < len(outputs); i++ {
        loss += (outputs[i] - targets[i]) * (outputs[i] - targets[i])
    }

    return loss
}

