# Neural Style Transfer

Transform regular photographs into stunning artistic masterpieces using deep learning!

## Overview

This project implements **Neural Style Transfer**, a technique that applies artistic styles from one image to the content of another using pre-trained convolutional neural networks. The implementation uses a VGG19 model to extract and combine content and style features, creating visually artistic results.

## Features

✨ **Pre-trained VGG19 Model** - No training required, ready to use out-of-the-box
🎨 **Artistic Style Application** - Transfer any artistic style to your photographs
⚡ **GPU Acceleration** - Automatic GPU support for faster processing
📊 **Loss Visualization** - Track content and style loss convergence
🔧 **Customizable Parameters** - Adjust style-to-content ratio for different effects
📈 **Advanced Techniques** - Explore different weight ratios and layer selections

## How It Works

The algorithm separates image content and style using deep convolutional features:

1. **Content Extraction** - Captures structural information from deeper layers
2. **Style Extraction** - Captures texture and patterns using Gram matrices
3. **Loss Computation** - Minimizes the difference between generated, content, and style images
4. **Optimization** - Uses Adam optimizer to iteratively refine the output

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Open `Neural_Style_Transfer.ipynb` in Jupyter Notebook or VS Code
2. Execute cells sequentially from top to bottom
3. Styled images will be saved to `style_transfer_results/` directory

### Usage Example

```python
# The notebook handles everything automatically:
# 1. Creates sample content and style images
# 2. Loads VGG19 model
# 3. Extracts features
# 4. Optimizes generated image
# 5. Saves and displays results
```

## Output Files

After running the notebook, you'll get:

- `01_original_images.png` - Content and style images comparison
- `02_loss_convergence.png` - Loss curves during optimization
- `03_final_comparison.png` - Side-by-side before/after comparison
- `04_weight_ratios_comparison.png` - Effect of different weight ratios
- `styled_output.jpg` - Final stylized image

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `iterations` | 200 | Number of optimization steps |
| `content_weight` | 1.0 | Emphasis on preserving content |
| `style_weight` | 100000.0 | Emphasis on applying style |
| `image_size` | 256 | Resolution of generated image |

## Performance Tips

- **GPU** - ~10-100x faster than CPU processing
- **Image Size** - Larger images take longer; 256x256 is recommended
- **Iterations** - 100-200 usually produces good results
- **Memory** - Reduce image size if out of memory

## Architecture

### Loss Functions

**Content Loss:** Measures difference in content features
```
L_content = MSE(generated_features, content_features)
```

**Style Loss:** Compares Gram matrices for texture similarity
```
L_style = Σ MSE(Gram(generated), Gram(style))
```

**Total Loss:** Weighted combination
```
L_total = α * L_content + β * L_style
```

### Feature Extraction

- **Content Layers:** relu4_2 (deep layer captures high-level content)
- **Style Layers:** relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 (multiple scales capture patterns)

## Real-World Applications

🖼️ Photo Artistic Rendering  
🎬 Video Frame Stylization  
🎮 Game Asset Generation  
🖌️ Creative Content Generation  
📱 Artistic Filter Applications  

## Advantages

✓ Works with any artistic style image  
✓ Creates high-quality artistic results  
✓ Interpretable loss functions  
✓ No neural network training required  
✓ Easy to experiment with parameters  

## Limitations

✗ Slow optimization (minutes per image on CPU)  
✗ Content/style weight tuning is manual  
✗ Requires similar image resolutions  
✗ GPU memory constraints for large images  

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- NumPy
- Matplotlib
- GPU (recommended, CPU supported)

## References

- Gatys et al. (2016) - "A Neural Algorithm of Artistic Style"
- VGG19: Very Deep Convolutional Networks for Large-Scale Image Recognition
- PyTorch Official Documentation

## License

MIT License - See LICENSE file for details

## Author

Created as a demonstration of Neural Style Transfer techniques using PyTorch and VGG19.

---

**Ready to create art? Run the notebook and transform your photos! 🎨**
