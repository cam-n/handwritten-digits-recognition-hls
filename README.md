# Handwritten Digits Recognition: Software and Hardware (LeNet-5, Python + FPGA)

A handwritten digit recognition application implemented in both software and hardware. The **LeNet-5** network is trained in Python (PyTorch) on the MNIST dataset, and the trained parameters are then used for **inference on an FPGA** (Zedboard), built with High-Level Synthesis (Vivado HLS).

## Overview

| Item | Description |
|---|---|
| Model | [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) |
| Application | Handwritten digit recognition |
| Dataset | [MNIST](http://yann.lecun.com/exdb/mnist/) |
| Training | Python (PyTorch), generates the network parameters |
| Inference | Python (software) and FPGA (hardware) |
| Board | Zedboard |

## Design Flow

```
 Python / PyTorch          C / HLS               Vivado HLS            Vivado               Zedboard
┌────────────────┐    ┌───────────────┐    ┌────────────────┐    ┌───────────────┐    ┌──────────────┐
│ Train LeNet-5  │ -> │ C inference   │ -> │ C to RTL       │ -> │ Build the SoC │ -> │ Run inference│
│ on MNIST       │    │ with trained  │    │ (IP core)      │    │ hardware      │    │ on FPGA      │
│ -> parameters  │    │ parameters    │    │                │    │ design        │    │              │
└────────────────┘    └───────────────┘    └────────────────┘    └───────────────┘    └──────────────┘
```

1. **Train** the network in Python to generate the weights and biases.
2. **Port** the inference phase to C/HLS and feed it the trained parameters. Verify it with a C compiler (GCC).
3. **Synthesize** the C/HLS program to RTL with Vivado HLS.
4. **Build** the hardware system around the generated IP core in Vivado.
5. **Run** inference on the Zedboard.

## Repository Structure

```
handwritten-digits-recognition-hls/
├── python/     # Training and software inference (PyTorch)
├── c_hls/      # C/HLS code for the inference phase
├── soc/        # Vivado hardware design (system on chip)
└── README.md
```

## Tools

| Tool | Purpose |
|---|---|
| Anaconda3 (Spyder, Python 3, PyTorch) | Training and software inference |
| Visual Studio Code (GCC) | Build and test the C/HLS program |
| Vivado HLS | Convert the C/HLS program to RTL |
| Vivado | Build the hardware for the inference phase |
| Zedboard | Target FPGA board |

## Getting Started

```bash
git clone https://github.com/cam-n/handwritten-digits-recognition-hls.git
cd handwritten-digits-recognition-hls
```

1. **Train.** Run the scripts in `python/` to train LeNet-5 on MNIST and export the parameters.
2. **Test the C/HLS code.** Build the program in `c_hls/` with GCC and check that its results match the Python inference.
3. **Synthesize.** Import `c_hls/` into Vivado HLS, run C synthesis, and export the RTL IP core.
4. **Build the hardware.** Open the design in `soc/` in Vivado, add the IP core, and generate the bitstream.
5. **Run on the board.** Program the Zedboard and run inference.

## Reference

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, 1998. [PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

## Author

[cam-n](https://github.com/cam-n)