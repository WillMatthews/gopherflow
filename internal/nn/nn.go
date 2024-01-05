package nn

import (
    "fmt"
    "github.com/WillMatthews/gopherflow/internal/engine"
)

type MLP struct {
    layers []*Layer
}

func NewMLP() *MLP {
    return &MLP{}
}

type Layer struct {
    neurons []*Neuron
}

type Neuron struct {
}



// nn activation functions

func relu(x float64) float64 {
    if x < 0 {
        return 0
    }
    return x
}

func leakyRelu(x float64) float64 {
    if x < 0 {
        return 0.01 * x
    }
    return x
}

func tanh(x float64) float64 {
    return (1 - x) / (1 + x)
}

func sigmoid(x float64) float64 {
    return 1 / (1 + x)
}

func softmax(x float64) float64 {
    return 1 / (1 + x)
}

func (m *MLP) AddLayer(n int, activation string) {
    l := &Layer{}
    for i := 0; i < n; i++ {
        l.neurons = append(l.neurons, &Neuron{})
    }
    m.layers = append(m.layers, l)
}
