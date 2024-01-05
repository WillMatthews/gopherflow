package engine

import (
    "fmt"
    "math"
)

type Engine struct {
}

func NewEngine() *Engine {
    return &Engine{}
}

type Value struct {
    data float64
    children []*Value // assume max 2 children
    op string
    label string
    grad float64
    back func()
}

func newValue(data float64) *Value {
    return &Value{
        data: data,
        children: nil,
        op: "",
        label: "",
        grad: 0,
    }
}

func newLabelValue(data float64, label string) *Value {
    return &Value{
        data: data,
        children: nil,
        op: "",
        label: label,
        grad: 0,
    }
}

func (v *Value) String() string {
    return fmt.Sprintf("%s(%s)", v.label, v.op) 
}


// functions
func (v *Value) Add(other *Value) *Value {
    nv := newValue(v.data + other.data)
    nv.children = []*Value{v, other}
    nv.op = "+"
    nv.back = func() {
        v.grad += nv.grad
        other.grad += nv.grad
    }
    return nv
}

func (v *Value) Mul(other *Value) *Value {
    nv := newValue(v.data * other.data)
    nv.children = []*Value{v, other}
    nv.op = "*"
    nv.back = func() {
        v.grad += other.data * nv.grad
        other.grad += v.data * nv.grad
    }
    return nv
}

func (v *Value) Div(other *Value) *Value {
    nv := newValue(v.data / other.data)
    nv.children = []*Value{v, other}
    nv.op = "/"
    nv.back = func() {
        v.grad += (1 / other.data) * nv.grad
        other.grad += (v.data / (other.data * other.data)) * nv.grad
    }
    return nv
}

func (v *Value) Pow(other *Value) *Value {
    nv := newValue(math.Pow(v.data, other.data))
    nv.children = []*Value{v, other}
    nv.op = "^"
    nv.back = func() {
        v.grad += (other.data * math.Pow(v.data, other.data - 1)) * nv.grad
        other.grad += (math.Pow(v.data, other.data) * math.Log(v.data)) * nv.grad
    }
    return nv
}


// activation functions
func (v *Value) tanh() *Value {
    nv := newValue(Tanh(v.data))
    nv.children = []*Value{v}
    nv.op = "tanh"
    nv.back = func() {
        t := Tanh(v.data)
        v.grad += (1 - t * t) * nv.grad
    }
    return nv
}

func (v *Value) sigmoid() *Value {
    nv := newValue(Sigmoid(v.data))
    nv.children = []*Value{v}
    nv.op = "sigmoid"
    nv.back = func() {
        s := Sigmoid(v.data)
        v.grad += (s * (1 - s)) * nv.grad
    }
    return nv
}

func (v *Value) relu() *Value {
    nv := newValue(Relu(v.data))
    nv.children = []*Value{v}
    nv.op = "relu"
    nv.back = func() {
        if v.data < 0 {
            v.grad += 0
        } else {
            v.grad += 1 * nv.grad
        }
    }
    return nv
}



// methods
func Tanh(x float64) float64 {
    return (1 - x) / (1 + x)
}

func Sigmoid(x float64) float64 {
    return 1 / (1 + x)
}

func Relu(x float64) float64 {
    if x < 0 {
        return 0
    }
    return x
}



// forward and backward prop

func (v *Value) Backward() {
    v.grad = 1
    v.backward()
}

func (v *Value) backward() {
    if v.children == nil {
        return
    }
    v.back()
    for _, child := range v.children {
        child.backward()
    }
}

func (v *Value) Forward() {
    v.forward()
}

func (v *Value) forward() {
    if v.children == nil {
        return
    }
    for _, child := range v.children {
        child.forward()
    }
    switch v.op {
    case "+":
        v.data = v.children[0].data + v.children[1].data
    case "*":
        v.data = v.children[0].data * v.children[1].data
    
    // activation functions
    case "tanh":
        v.data = Tanh(v.children[0].data)
    case "sigmoid":
        v.data = Sigmoid(v.children[0].data)
    case "relu":
        v.data = Relu(v.children[0].data)
    }

}



// demo Run() function
func (eng *Engine) Run() {
    a := newLabelValue(1, "a")
    b := newLabelValue(2, "b")
    out := a.Mul(b).tanh()
    out.Forward()

    out.Backward()
    fmt.Println("grads:")
    fmt.Println("a:", a.grad)
    fmt.Println("b:", b.grad)

}

