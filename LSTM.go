package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
)

// Sigmoid activation function and its derivative
func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
    return x * (1.0 - x)
}

// Tanh activation function and its derivative
func tanh(x float64) float64 {
    return math.Tanh(x)
}

func tanhDerivative(x float64) float64 {
    return 1.0 - (x * x)
}

// LSTM Cell struct
type LSTMCell struct {
    inputSize  int
    hiddenSize int

    // Weight matrices
    Wf, Wi, Wo, Wc [][]float64
    Uf, Ui, Uo, Uc [][]float64
    bf, bi, bo, bc []float64

    // Cell states
    h, c []float64
}

// Create a new LSTM cell
func NewLSTMCell(inputSize, hiddenSize int) *LSTMCell {
    cell := &LSTMCell{
        inputSize:  inputSize,
        hiddenSize: hiddenSize,
    }

    // Initialize weight matrices
    cell.Wf = randomMatrix(hiddenSize, inputSize)
    cell.Wi = randomMatrix(hiddenSize, inputSize)
    cell.Wo = randomMatrix(hiddenSize, inputSize)
    cell.Wc = randomMatrix(hiddenSize, inputSize)

    cell.Uf = randomMatrix(hiddenSize, hiddenSize)
    cell.Ui = randomMatrix(hiddenSize, hiddenSize)
    cell.Uo = randomMatrix(hiddenSize, hiddenSize)
    cell.Uc = randomMatrix(hiddenSize, hiddenSize)

    cell.bf = randomVector(hiddenSize)
    cell.bi = randomVector(hiddenSize)
    cell.bo = randomVector(hiddenSize)
    cell.bc = randomVector(hiddenSize)

    // Initialize cell states
    cell.h = make([]float64, hiddenSize)
    cell.c = make([]float64, hiddenSize)

    return cell
}

// Random matrix initialization
func randomMatrix(rows, cols int) [][]float64 {
    matrix := make([][]float64, rows)
    for i := range matrix {
        matrix[i] = make([]float64, cols)
        for j := range matrix[i] {
            matrix[i][j] = rand.Float64()
        }
    }
    return matrix
}

// Random vector initialization
func randomVector(size int) []float64 {
    vector := make([]float64, size)
    for i := range vector {
        vector[i] = rand.Float64()
    }
    return vector
}

// Forward pass through the LSTM cell
func (cell *LSTMCell) Forward(input []float64) []float64 {
    // Calculate gates and cell state
    ft := sigmoid(matrixVectorAdd(matrixVectorMul(cell.Wf, input), matrixVectorMul(cell.Uf, cell.h), cell.bf))
    it := sigmoid(matrixVectorAdd(matrixVectorMul(cell.Wi, input), matrixVectorMul(cell.Ui, cell.h), cell.bi))
    ot := sigmoid(matrixVectorAdd(matrixVectorMul(cell.Wo, input), matrixVectorMul(cell.Uo, cell.h), cell.bo))
    ctCandidate := tanh(matrixVectorAdd(matrixVectorMul(cell.Wc, input), matrixVectorMul(cell.Uc, cell.h), cell.bc))

    cell.c = vectorAdd(vectorMul(ft, cell.c), vectorMul(it, ctCandidate))
    cell.h = vectorMul(ot, tanh(cell.c))

    return cell.h
}

// Matrix-vector multiplication
func matrixVectorMul(matrix [][]float64, vector []float64) []float64 {
    result := make([]float64, len(matrix))
    for i := range matrix {
        for j := range matrix[i] {
            result[i] += matrix[i][j] * vector[j]
        }
    }
    return result
}

// Matrix-vector addition
func matrixVectorAdd(vectors ...[]float64) []float64 {
    result := make([]float64, len(vectors[0]))
    for _, vector := range vectors {
        for i := range vector {
            result[i] += vector[i]
        }
    }
    return result
}

// Element-wise vector addition
func vectorAdd(a, b []float64) []float64 {
    result := make([]float64, len(a))
    for i := range a {
        result[i] = a[i] + b[i]
    }
    return result
}

// Element-wise vector multiplication
func vectorMul(a, b []float64) []float64 {
    result := make([]float64, len(a))
    for i := range a {
        result[i] = a[i] * b[i]
    }
    return result
}

// Neural Network struct with LSTM
type NeuralNetwork struct {
    lstmCells []*LSTMCell
    inputSize int
    hiddenSize int
    outputSize int

    // Weight matrices
    Wout [][]float64
    bout []float64
}

// Create a new neural network with LSTM layers
func NewNeuralNetwork(inputSize, hiddenSize, outputSize, numLayers int) *NeuralNetwork {
    nn := &NeuralNetwork{
        inputSize:  inputSize,
        hiddenSize: hiddenSize,
        outputSize: outputSize,
    }

    // Initialize LSTM cells
    for i := 0; i < numLayers; i++ {
        nn.lstmCells = append(nn.lstmCells, NewLSTMCell(inputSize, hiddenSize))
        inputSize = hiddenSize // Output of the previous layer is input to the next layer
    }

    // Initialize output weight matrix
    nn.Wout = randomMatrix(outputSize, hiddenSize)
    nn.bout = randomVector(outputSize)

    return nn
}

// Forward pass through the neural network
func (nn *NeuralNetwork) Forward(input []float64) []float64 {
    // Pass input through LSTM layers
    for _, cell := range nn.lstmCells {
        input = cell.Forward(input)
    }

    // Output layer
    output := make([]float64, nn.outputSize)
    for i := range nn.Wout {
        for j := range nn.Wout[i] {
            output[i] += nn.Wout[i][j] * input[j]
        }
        output[i] += nn.bout[i]
        output[i] = sigmoid(output[i]) // Apply activation function
    }

    return output
}

func main() {
    rand.Seed(time.Now().UnixNano())

    // Create a neural network with 3 input nodes, 5 hidden nodes, and 2 output nodes with 2 LSTM layers
    nn := NewNeuralNetwork(3, 5, 2, 2)

    // Sample input sequence
    input := []float64{0.5, 0.2, 0.8}

    // Forward pass through the network
    output := nn.Forward(input)

    fmt.Printf("Input: %v, Output: %v\n", input, output)
}
