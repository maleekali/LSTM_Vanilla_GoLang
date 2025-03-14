
```markdown
# Complex RNN with LSTM in Vanilla Go

This repository contains a complex Recurrent Neural Network (RNN) implementation with Long Short-Term Memory (LSTM) units, written entirely in vanilla Go. The project demonstrates a sequence prediction task using LSTM cells without relying on any external machine learning frameworks.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Environment Setup](#environment-setup)
- [LSTM Cell Implementation](#lstm-cell-implementation)
- [Neural Network with LSTM Layers](#neural-network-with-lstm-layers)
- [Main Function](#main-function)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project showcases a complex RNN with LSTM units implemented from scratch using vanilla Go. The neural network is designed for sequence prediction tasks and demonstrates the inner workings of LSTM cells, including the input gate, forget gate, output gate, and cell state calculations.

## Objectives
- Implement LSTM cells from scratch in Go.
- Build a neural network with multiple LSTM layers.
- Perform sequence prediction using the neural network.

## Environment Setup
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/complex-rnn-lstm-go.git
   cd complex-rnn-lstm-go
   ```

2. Ensure you have Go installed on your system. If not, you can download and install it from the [official Go website](https://golang.org/dl/).

## LSTM Cell Implementation
The LSTM cell is implemented with the following components:
- Input gate
- Forget gate
- Output gate
- Cell state calculations

### Example LSTM Cell Code
```go
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
```

## Neural Network with LSTM Layers
The neural network is built with multiple LSTM layers. Each layer processes the input sequence and passes the output to the next layer.

### Example Neural Network Code
```go
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
```

## Main Function
The main function demonstrates how to create the neural network, perform a forward pass with a sample input sequence, and print the results.

### Example Main Function Code
```go
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
```

## Usage
1. Compile and run the Go program:
   ```bash
   go run main.go
   ```

2. The program will print the output for the sample input sequence.