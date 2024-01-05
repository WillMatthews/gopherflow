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
    return nv
}

func (v *Value) Mul(other *Value) *Value {
    nv := newValue(v.data * other.data)
    nv.children = []*Value{v, other}
    nv.op = "*"
    return nv
}

func (v *Value) Div(other *Value) *Value {
    nv := newValue(v.data / other.data)
    nv.children = []*Value{v, other}
    nv.op = "/"
    return nv
}

func (v *Value) Pow(other *Value) *Value {
    nv := newValue(math.Pow(v.data, other.data))
    nv.children = []*Value{v, other}
    nv.op = "^"
    return nv
}


// activation functions
func (v *Value) tanh() *Value {
    nv := newValue(tanh(v.data))
    nv.children = []*Value{v}
    nv.op = "tanh"
    return nv
}

func (v *Value) sigmoid() *Value {
    nv := newValue(sigmoid(v.data))
    nv.children = []*Value{v}
    nv.op = "sigmoid"
    return nv
}

func (v *Value) relu() *Value {
    nv := newValue(relu(v.data))
    nv.children = []*Value{v}
    nv.op = "relu"
    return nv
}



// methods
func tanh(x float64) float64 {
    return (1 - x) / (1 + x)
}

func sigmoid(x float64) float64 {
    return 1 / (1 + x)
}

func relu(x float64) float64 {
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

    for i, child := range v.children {
        switch v.op {
        case "+":
            child.grad += v.grad
        case "*":
            // Assume there are only two children..?
            child.grad += v.grad * v.children[(i+1)%2].data
        case "tanh":
            child.grad += v.grad * (1 - v.data * v.data)
        case "sigmoid":
            child.grad += v.grad * v.data * (1 - v.data)
        case "relu":
            if v.data < 0 {
                child.grad += 0
            } else {
                child.grad += v.grad
            }
        }
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
        v.data = tanh(v.children[0].data)
    case "sigmoid":
        v.data = sigmoid(v.children[0].data)
    case "relu":
        v.data = relu(v.children[0].data)
    }

}



// demo Run() function
func (eng *Engine) Run() {
    var a, b, c, d, e, f *Value
    a = newLabelValue(1, "a")
    b = newLabelValue(2, "b")
    c = newLabelValue(3, "c")
    d = a.Add(b) // 1 + 2
    e = d.Mul(c) // (1 + 2) * 3
    f = e.Add(d) // (1 + 2) * 3 + (1 + 2)
    out := f.tanh()
    
    // print out the tree
    fmt.Println(out)
    fmt.Println(f.children[0])
    fmt.Println(f.children[1])
    fmt.Println(f.children[0].children[0])
    fmt.Println(f.children[0].children[1])
    fmt.Println(f.children[1].children[0])
    fmt.Println(f.children[1].children[1])

    // forward pass
    out.Forward()
    fmt.Println("forward pass")
    fmt.Println(f.data)
    fmt.Println(f.children[0].data)
    fmt.Println(f.children[1].data)
    fmt.Println(f.children[0].children[0].data)
    fmt.Println(f.children[0].children[1].data)
    fmt.Println(f.children[1].children[0].data)
    fmt.Println(f.children[1].children[1].data)

    // backward pass
    out.Backward()
    fmt.Println("backward pass")
    fmt.Println(f.grad)
    fmt.Println(f.children[0].grad)
    fmt.Println(f.children[1].grad)
    fmt.Println(f.children[0].children[0].grad)
    fmt.Println(f.children[0].children[1].grad)
    fmt.Println(f.children[1].children[0].grad)
    fmt.Println(f.children[1].children[1].grad)


}

