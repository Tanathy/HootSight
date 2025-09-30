# Model Architecture Guide

This guide covers all neural network architectures supported by Hootsight, their available variants, characteristics, and use cases. Hootsight supports six major architecture families with a total of 28 specific model variants.

## Supported Architecture Families

Hootsight implements six distinct neural network architecture families, each optimized for different scenarios:

1. **ResNet** - Deep residual networks with skip connections
2. **ResNeXt** - Enhanced ResNet with grouped convolutions  
3. **MobileNet** - Lightweight networks for mobile and edge devices
4. **ShuffleNet** - Efficient networks with channel shuffling
5. **SqueezeNet** - Compact networks with fire modules
6. **EfficientNet** - Compound-scaled networks optimizing accuracy and efficiency

## Architecture Details

### ResNet (Residual Networks)

**Purpose**: Revolutionary deep convolutional networks that solve the vanishing gradient problem through residual skip connections, enabling training of unprecedented network depths.

**Historical Significance**: ResNet was introduced in the groundbreaking paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. It won the 2015 ILSVRC & COCO competition and achieved a milestone 3.57% error on ImageNet, representing a 28% relative improvement in object detection tasks.

**Core Innovation**: ResNet introduced residual connections that allow networks to learn residual functions with reference to layer inputs, rather than learning unreferenced functions. This breakthrough enables training networks with up to 1000 layers while remaining easier to optimize than shallower alternatives.

**Available Variants**:
- `resnet18` - 18 layers, 11.7M parameters (basic blocks)
- `resnet34` - 34 layers, 21.8M parameters (basic blocks)
- `resnet50` - 50 layers, 25.6M parameters (bottleneck blocks)
- `resnet101` - 101 layers, 44.5M parameters (bottleneck blocks)
- `resnet152` - 152 layers, 60.2M parameters (bottleneck blocks)

**Architecture Details**:
- **Residual Blocks**: The fundamental building block uses skip connections that add the input directly to the output of stacked layers
- **Basic vs Bottleneck**: ResNet-18/34 use basic blocks (two 3×3 convolutions), while ResNet-50+ use bottleneck blocks (1×1, 3×3, 1×1 convolutions)
- **Skip Connections**: Enable gradient flow directly to earlier layers, solving vanishing gradients in very deep networks
- **Implementation**: Hootsight wraps torchvision's ResNet family and configures task-specific heads for classification, multi-label, detection, and segmentation scenarios

**Technical Characteristics**:
- **Architecture**: Residual learning with identity shortcut connections
- **Strengths**: Exceptional depth capability, excellent gradient flow, proven performance across tasks
- **Performance**: Optimal balance of accuracy, training stability, and computational efficiency
- **Memory Usage**: Moderate to high depending on variant and input resolution
- **Training Speed**: Fast to moderate depending on depth, very stable convergence

**Task Support**:
- ✅ **Classification**: Full support with linear classifier head on pooled features
- ✅ **Multi-label**: Sigmoid activation for multiple simultaneous class predictions
- ✅ **Detection**: Faster R-CNN with ResNet-50 FPN backbone (optimized for object detection)
- ✅ **Segmentation**: DeepLabv3 with ResNet-50 backbone (optimized for semantic segmentation)

**Architecture Scaling**: ResNet demonstrates that network depth is critically important for visual recognition tasks. The residual learning framework allows networks to be substantially deeper (8× deeper than VGG) while maintaining lower complexity and achieving superior accuracy.

**Block Structure Differences**:
- **ResNet-18/34**: Use basic residual blocks with two 3×3 convolutional layers each
- **ResNet-50/101/152**: Use bottleneck blocks with 1×1, 3×3, 1×1 convolutions for computational efficiency

**Why ResNet Works**:
1. **Gradient Flow**: Skip connections provide direct paths for gradients, preventing vanishing gradients
2. **Identity Mapping**: Layers can learn to approximate identity functions when beneficial
3. **Feature Reuse**: Lower-level features are preserved and combined with higher-level features
4. **Training Stability**: Residual learning is easier to optimize than learning unreferenced functions

**When to Use**:
- **General Purpose**: ResNet-50 is the gold standard starting point for most computer vision tasks
- **High Accuracy**: ResNet-101/152 when maximum accuracy is needed and computational resources allow
- **Fast Experimentation**: ResNet-18/34 for rapid prototyping and proof-of-concept work
- **Detection/Segmentation**: ResNet-50 provides the optimal backbone for specialized task heads
- **Transfer Learning**: Exceptional pretrained weights from ImageNet for most visual domains
- **Research Baseline**: Widely accepted baseline for comparing new architectures and techniques

**Configuration Example**:
```json
{
  "training": {
    "model_type": "resnet",
    "model_name": "resnet50",
    "pretrained": true,
    "task": "classification"
  }
}
```

**Legacy and Impact**: ResNet's introduction marked a fundamental shift in deep learning architecture design. Its residual learning principle has been adopted across countless subsequent architectures, making it one of the most influential contributions to computer vision. The ability to train networks with 152+ layers opened new possibilities for representation learning and established the foundation for modern deep learning in computer vision.

### ResNeXt (Aggregated Residual Transformations)

**Purpose**: Enhanced version of ResNet using grouped convolutions for better accuracy with similar complexity.

**Available Variants**:
- `resnext50_32x4d` - 50 layers, 32 groups, 4d width, 25.0M parameters
- `resnext101_32x8d` - 101 layers, 32 groups, 8d width, 88.8M parameters
- `resnext101_64x4d` - 101 layers, 64 groups, 4d width, 83.5M parameters

**Characteristics**:
- **Architecture**: Grouped convolutions that split channels into parallel paths
- **Strengths**: Better accuracy than ResNet at similar computational cost
- **Performance**: Higher accuracy than equivalent ResNet models
- **Memory Usage**: Similar to ResNet but with more parameters
- **Training Speed**: Slightly slower than ResNet due to grouped operations

**Task Support**:
- ✅ **Classification**: Full support with excellent accuracy
- ✅ **Multi-label**: Optimized for multi-class scenarios
- ❌ **Detection**: Classification only (use ResNet for detection)
- ❌ **Segmentation**: Classification only (use ResNet for segmentation)

**When to Use**:
- **Higher accuracy needed**: When ResNet accuracy is insufficient
- **Multi-class problems**: Particularly effective for complex classification
- **Sufficient resources**: When you can afford slightly higher computational cost
- **Production deployments**: When accuracy is more important than speed

### MobileNet (Mobile Networks)

**Purpose**: Pioneering family of lightweight neural networks specifically designed for mobile and edge devices with limited computational resources, introducing revolutionary efficiency techniques for practical deployment.

**Historical Significance**: MobileNets introduced the foundational concept of depthwise separable convolutions for efficient neural networks. Each version represents significant architectural innovations: V1 established depthwise separable convolutions, V2 introduced inverted residuals with linear bottlenecks, and V3 incorporated neural architecture search with hardware-aware optimization.

**Core Innovations**:
- **V1**: Depthwise separable convolutions replacing standard convolutions
- **V2**: Inverted residual blocks with linear bottlenecks for better gradient flow  
- **V3**: Neural architecture search (NAS) optimization with hardware-aware design and h-swish activation

**Available Variants**:
- `mobilenet_v2` - Inverted residuals with linear bottlenecks, 3.5M parameters
- `mobilenet_v3_small` - NAS-optimized for mobile CPUs, 2.5M parameters, minimal latency
- `mobilenet_v3_large` - NAS-optimized for balanced performance, 5.5M parameters, optimal accuracy/efficiency

**Architecture Evolution**:

**MobileNet V1** (Foundation):
- **Depthwise Separable Convolutions**: Factorizes standard convolution into depthwise convolution followed by 1×1 pointwise convolution
- **Computational Reduction**: Reduces computation by factor of 1/N + 1/D² where N is number of output channels and D is kernel size
- **Width Multiplier**: α parameter to uniformly thin networks at each layer

**MobileNet V2** (Inverted Residuals):
- **Inverted Residual Structure**: Expand → depthwise → project pattern (opposite of traditional residuals)
- **Linear Bottlenecks**: Remove non-linearities from narrow layers to preserve information
- **Expansion Factor**: t parameter (typically 6) controls intermediate expansion size
- **Skip Connections**: Added between bottleneck layers for gradient flow

**MobileNet V3** (NAS Optimization):
- **Platform-Aware NAS**: Neural architecture search optimized for mobile hardware performance
- **SE (Squeeze-and-Excite) Modules**: Selective application of attention mechanisms
- **h-swish Activation**: Hardware-friendly version of swish activation function
- **Redesigned Head**: Efficient last stage reducing latency by 15% with minimal accuracy loss

**Technical Characteristics**:
- **Architecture**: Depthwise separable convolutions with generation-specific optimizations
- **Strengths**: Exceptional inference speed, minimal model size, ultra-low memory footprint, mobile-optimized
- **Performance**: Outstanding accuracy-to-size ratio, optimized for real-world mobile deployment scenarios
- **Memory Usage**: Extremely low, suitable for devices with <1GB RAM
- **Training Speed**: Very fast due to reduced computational complexity

**Task Support**:
- ✅ **Classification**: Optimized for mobile and edge classification with excellent efficiency
- ✅ **Multi-label**: Efficient multi-class classification suitable for resource-constrained scenarios
- ❌ **Detection**: Classification focus only (use ResNet for object detection)
- ❌ **Segmentation**: Classification only (use ResNet for semantic segmentation)

**Depthwise Separable Convolution Benefits**:
1. **Computational Efficiency**: 8-9× reduction in computation compared to standard convolutions
2. **Parameter Efficiency**: Dramatic reduction in model size while maintaining representational power
3. **Memory Efficiency**: Lower activation memory requirements for mobile deployment
4. **Hardware Optimization**: Designed to leverage mobile GPU and CPU capabilities efficiently

**Version Selection Guide**:
- **MobileNet V2**: Proven architecture with excellent balance of accuracy and efficiency, widely supported
- **MobileNet V3 Small**: Ultra-lightweight for severely constrained devices, optimized for minimal latency
- **MobileNet V3 Large**: Best MobileNet accuracy while maintaining efficiency, NAS-optimized architecture

**When to Use**:
- **Mobile Applications**: iOS and Android apps requiring on-device inference
- **Edge Computing**: IoT devices, embedded systems, and edge servers with limited resources
- **Real-Time Inference**: Applications where inference speed is critical (video processing, live classification)
- **Battery-Powered Devices**: Systems where power consumption directly impacts battery life
- **Network-Constrained Environments**: Scenarios requiring model downloads over limited bandwidth
- **Rapid Prototyping**: Quick experimentation with fast training and deployment cycles
- **Cost-Sensitive Deployment**: Cloud deployment where computational costs need optimization

**Hardware Optimization**:
- **Mobile GPUs**: Optimized for ARM Mali, Adreno, and Apple GPU architectures
- **Mobile CPUs**: Efficient utilization of ARM Cortex and Apple A-series processors
- **Specialized Chips**: Compatible with mobile neural processing units and DSPs
- **Quantization Friendly**: Architecture supports INT8 quantization with minimal accuracy loss

**Configuration Example**:
```json
{
  "training": {
    "model_type": "mobilenet",
    "model_name": "mobilenet_v3_large", 
    "pretrained": true,
    "input_size": 224
  }
}
```

**Impact and Legacy**: MobileNets revolutionized on-device AI by proving that careful architectural design could achieve practical accuracy with mobile-friendly efficiency. The depthwise separable convolution concept has been adopted across numerous efficient architectures, establishing MobileNet as the foundation for mobile-first neural network design.

### ShuffleNet (Channel Shuffle Networks)

**Purpose**: Extremely efficient networks using channel shuffling to reduce computational cost while maintaining accuracy.

**Available Variants**:
- `shufflenet_v2_x0_5` - 0.5x width multiplier, 1.4M parameters
- `shufflenet_v2_x1_0` - 1.0x width multiplier, 2.3M parameters
- `shufflenet_v2_x1_5` - 1.5x width multiplier, 3.5M parameters
- `shufflenet_v2_x2_0` - 2.0x width multiplier, 7.4M parameters

**Characteristics**:
- **Architecture**: Channel shuffling with grouped convolutions
- **Strengths**: Exceptional efficiency, very low computational cost
- **Performance**: Good accuracy for extremely small models
- **Memory Usage**: Minimal
- **Training Speed**: Very fast

**Task Support**:
- ✅ **Classification**: Highly efficient classification
- ✅ **Multi-label**: Suitable for lightweight multi-class tasks
- ❌ **Detection**: Classification only
- ❌ **Segmentation**: Classification only

**When to Use**:
- **Extreme efficiency**: When computational resources are severely limited
- **Battery-powered devices**: Devices where power consumption matters
- **High-throughput inference**: Processing many images quickly
- **Embedded systems**: Microcontrollers and similar constrained hardware
- **Educational purposes**: Understanding efficient network design

**Width Multiplier Guide**:
- **x0.5**: Absolute minimum resources, basic classification
- **x1.0**: Good balance for most mobile applications  
- **x1.5**: Better accuracy when resources allow
- **x2.0**: Best ShuffleNet accuracy, still very efficient

### SqueezeNet (Fire Module Networks)

**Purpose**: Compact networks using fire modules to achieve good accuracy with minimal parameters.

**Available Variants**:
- `squeezenet1_0` - Original SqueezeNet architecture, 1.2M parameters
- `squeezenet1_1` - Improved version with better accuracy, 1.2M parameters

**Characteristics**:
- **Architecture**: Fire modules with squeeze and expand layers
- **Strengths**: Very small model size, good compression ratio
- **Performance**: Decent accuracy with minimal storage requirements
- **Memory Usage**: Very low
- **Training Speed**: Fast

**Task Support**:
- ✅ **Classification**: Compact classification models
- ✅ **Multi-label**: Efficient for simple multi-class tasks
- ❌ **Detection**: Classification only
- ❌ **Segmentation**: Classification only

**When to Use**:
- **Storage constraints**: When model file size is critical
- **Legacy hardware**: Older devices with limited capabilities
- **Network transmission**: Models that need to be downloaded frequently
- **Teaching/Research**: Understanding network compression techniques
- **Proof of concept**: Quick validation of ideas with minimal resources

**Version Differences**:
- **SqueezeNet 1.0**: Original design, good baseline
- **SqueezeNet 1.1**: Improved architecture with better accuracy/efficiency trade-off

### EfficientNet (Compound Scaling Networks)

**Purpose**: Revolutionary neural networks that systematically scale network dimensions using compound scaling for optimal accuracy-efficiency trade-offs, achieving state-of-the-art performance with significantly fewer parameters.

**Historical Significance**: EfficientNet was introduced in "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" by Mingxing Tan and Quoc V. Le. It achieved state-of-the-art 84.3% top-1 accuracy on ImageNet while being 8.4× smaller and 6.1× faster than the best existing ConvNet at the time.

**Core Innovation**: EfficientNet introduces compound scaling that uniformly scales all dimensions of depth, width, and resolution using a simple yet highly effective compound coefficient. This systematic approach to model scaling identifies that carefully balancing network depth, width, and resolution leads to superior performance compared to scaling individual dimensions.

**Available Variants**:

**EfficientNet V1** (Original Compound Scaling):
- `efficientnet_b0` - Base model, 5.3M parameters, 224×224 input
- `efficientnet_b1` - 1.2× scaled, 7.8M parameters, 240×240 input
- `efficientnet_b2` - 1.4× scaled, 9.2M parameters, 260×260 input  
- `efficientnet_b3` - 1.8× scaled, 12M parameters, 300×300 input
- `efficientnet_b4` - 2.2× scaled, 19M parameters, 380×380 input
- `efficientnet_b5` - 2.6× scaled, 30M parameters, 456×456 input
- `efficientnet_b6` - 3.1× scaled, 43M parameters, 528×528 input
- `efficientnet_b7` - 3.8× scaled, 66M parameters, 600×600 input

**EfficientNet V2** (Enhanced Training and Architecture):
- `efficientnet_v2_s` - Small V2, 22M parameters, improved training efficiency
- `efficientnet_v2_m` - Medium V2, 54M parameters, faster training convergence
- `efficientnet_v2_l` - Large V2, 119M parameters, maximum accuracy with better training

**Architecture Details**:
- **Neural Architecture Search**: EfficientNet-B0 baseline was designed using neural architecture search (NAS) for optimal efficiency
- **Compound Scaling Formula**: Uses compound coefficient φ to scale depth (α^φ), width (β^φ), and resolution (γ^φ) with α·β²·γ² ≈ 2
- **MBConv Blocks**: Built on Mobile Inverted Bottleneck Convolution (MBConv) blocks with squeeze-and-excitation optimization
- **Depth/Width/Resolution Balance**: Systematically balances all three dimensions rather than arbitrary scaling

**Technical Characteristics**:
- **Architecture**: Compound-scaled MBConv blocks with squeeze-and-excitation attention
- **Strengths**: Exceptional accuracy-to-efficiency ratio, superior transfer learning, systematic scaling approach
- **Performance**: Achieves highest accuracy among all supported architectures with optimal parameter efficiency
- **Memory Usage**: Scales predictably with variant size (B0: low, B7/V2-L: very high)
- **Training Speed**: Faster convergence than equivalent accuracy networks, V2 variants improve training efficiency further

**Compound Scaling Methodology**:
1. **Baseline Network**: Start with efficient baseline architecture (EfficientNet-B0)
2. **Scaling Constraints**: Fix compound coefficient φ=1 and search for optimal α, β, γ values
3. **Compound Scaling**: Scale up baseline using fixed ratios with larger φ values
4. **Resource Allocation**: Systematically distribute computational budget across depth, width, and resolution

**Task Support**:
- ✅ **Classification**: State-of-the-art performance for image classification with exceptional transfer learning
- ✅ **Multi-label**: Superior performance for complex multi-class scenarios, excellent for fine-grained classification
- ❌ **Detection**: Classification only (use ResNet backbone for object detection tasks)
- ❌ **Segmentation**: Classification only (use ResNet backbone for semantic segmentation)

**Transfer Learning Excellence**: EfficientNets achieve outstanding transfer learning results, obtaining state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and multiple other datasets with an order of magnitude fewer parameters than traditional approaches.

**Scaling Philosophy**: EfficientNet demonstrates that conventional practice of scaling ConvNets at fixed resource budgets is suboptimal. The compound scaling method shows that balancing all dimensions of network scaling leads to consistently better performance across different resource constraints.

**When to Use**:
- **Maximum Accuracy Requirements**: When achieving the highest possible classification accuracy is critical
- **Resource-Constrained High Performance**: Need excellent accuracy but with computational efficiency constraints  
- **Transfer Learning Projects**: Exceptional pretrained representations for domain adaptation
- **Production Systems**: High-quality applications where accuracy directly impacts business value
- **Competitive Scenarios**: Kaggle competitions and research benchmarks requiring state-of-the-art results
- **Commercial Applications**: Products where classification accuracy provides competitive advantage
- **Fine-Grained Classification**: Tasks requiring subtle visual distinction capabilities

**Scaling Strategy**:
- **B0-B1**: Excellent starting points with reasonable resource requirements, good for prototyping
- **B2-B3**: Balanced accuracy and efficiency, suitable for most production applications
- **B4-B5**: High accuracy with manageable computational cost for server deployment
- **B6-B7**: Maximum accuracy, requires significant computational resources, research and competition use
- **V2 Series**: Improved training efficiency and faster convergence than V1 equivalents, better for large-scale training

**Configuration Example**:
```json
{
  "training": {
    "model_type": "efficientnet", 
    "model_name": "efficientnet_b3",
    "pretrained": true,
    "input_size": 300
  }
}
```

**Performance Achievements**: EfficientNet-B7's 84.3% ImageNet top-1 accuracy represented a significant breakthrough, demonstrating that systematic scaling approaches can achieve superior results compared to ad-hoc architecture modifications. The compound scaling principle has influenced numerous subsequent architecture designs.

## Task Compatibility Matrix

| Architecture | Classification | Multi-label | Detection | Segmentation |
|-------------|---------------|-------------|-----------|-------------|
| ResNet | ✅ | ✅ | ✅ | ✅ |
| ResNeXt | ✅ | ✅ | ❌ | ❌ |
| MobileNet | ✅ | ✅ | ❌ | ❌ |
| ShuffleNet | ✅ | ✅ | ❌ | ❌ |
| SqueezeNet | ✅ | ✅ | ❌ | ❌ |
| EfficientNet | ✅ | ✅ | ❌ | ❌ |

**Note**: For detection and segmentation tasks, ResNet provides specialized implementations (Faster R-CNN and DeepLabv3 respectively) with ResNet-50 as the recommended backbone.

## Performance Characteristics

### Model Size Comparison (Parameters)

**Lightweight Models** (< 10M parameters):
- SqueezeNet 1.0/1.1: ~1.2M
- ShuffleNet V2 x0.5: ~1.4M  
- ShuffleNet V2 x1.0: ~2.3M
- MobileNet V3 Small: ~2.5M
- MobileNet V2: ~3.5M
- EfficientNet B0: ~5.3M
- ShuffleNet V2 x2.0: ~7.4M
- EfficientNet B1: ~7.8M

**Medium Models** (10M - 50M parameters):
- EfficientNet B2: ~9.2M
- ResNet-18: ~11.7M
- EfficientNet B3: ~12M
- EfficientNet B4: ~19M
- ResNet-34: ~21.8M
- EfficientNet V2 S: ~22M
- ResNet-50: ~25.6M
- ResNeXt-50: ~25.0M
- EfficientNet B5: ~30M
- ResNet-101: ~44.5M
- EfficientNet B6: ~43M

**Large Models** (> 50M parameters):
- EfficientNet V2 M: ~54M
- ResNet-152: ~60.2M  
- EfficientNet B7: ~66M
- ResNeXt-101 64x4d: ~83.5M
- ResNeXt-101 32x8d: ~88.8M
- EfficientNet V2 L: ~119M

### Speed vs Accuracy Trade-offs

**High Speed, Lower Accuracy**:
1. ShuffleNet V2 (all variants)
2. SqueezeNet 1.0/1.1
3. MobileNet V3 Small
4. MobileNet V2/V3 Large

**Balanced Speed and Accuracy**:
1. EfficientNet B0-B2
2. ResNet-18/34
3. MobileNet V3 Large
4. ResNeXt-50

**High Accuracy, Lower Speed**:
1. EfficientNet B5-B7
2. EfficientNet V2 M/L
3. ResNeXt-101 variants
4. ResNet-101/152

### Memory Usage Guidelines

**Low Memory** (< 4GB GPU):
- ShuffleNet V2 x0.5/x1.0
- SqueezeNet 1.0/1.1
- MobileNet V3 Small
- EfficientNet B0 (with batch size ≤ 16)

**Medium Memory** (4-8GB GPU):
- All MobileNet variants
- ShuffleNet V2 x1.5/x2.0
- ResNet-18/34
- EfficientNet B0-B2
- ResNeXt-50

**High Memory** (8-16GB GPU):
- ResNet-50/101
- EfficientNet B3-B5
- EfficientNet V2 S/M
- ResNeXt-101 variants

**Very High Memory** (>16GB GPU):
- ResNet-152
- EfficientNet B6/B7
- EfficientNet V2 L

## Selection Guidelines

### By Use Case

**Mobile Applications**:
- **Primary**: MobileNet V3 Large
- **Alternative**: MobileNet V2, ShuffleNet V2 x1.0
- **Minimal**: MobileNet V3 Small, ShuffleNet V2 x0.5

**Edge Devices/IoT**:
- **Primary**: ShuffleNet V2 x1.0
- **Alternative**: SqueezeNet 1.1, MobileNet V3 Small
- **Minimal**: ShuffleNet V2 x0.5

**Cloud/Server Deployment**:
- **Primary**: EfficientNet B3-B5
- **Alternative**: ResNet-50/101, ResNeXt-50
- **Maximum accuracy**: EfficientNet B7, EfficientNet V2 L

**Research/Competition**:
- **Primary**: EfficientNet V2 series
- **Alternative**: EfficientNet B5-B7
- **Baseline**: ResNet-50, EfficientNet B0

**Production Systems**:
- **Balanced**: EfficientNet B2-B4, ResNet-50
- **High throughput**: MobileNet V3 Large, ShuffleNet V2 x1.5
- **High accuracy**: EfficientNet B5+, ResNeXt-101

### By Dataset Size

**Small Datasets** (< 1,000 images):
- Lightweight models with strong regularization
- MobileNet V3 Small/Large, EfficientNet B0-B1
- Focus on transfer learning with pretrained weights

**Medium Datasets** (1,000 - 10,000 images):
- Balanced models with good generalization
- ResNet-34/50, EfficientNet B2-B3, MobileNet V3 Large
- Standard training procedures work well

**Large Datasets** (> 10,000 images):
- Any architecture suitable for your accuracy/speed requirements
- Larger models (ResNet-101, EfficientNet B4+) become viable
- Can train from scratch if needed

### By Accuracy Requirements

**Basic Accuracy** (Good enough for many applications):
- MobileNet V2/V3, ShuffleNet V2 x1.0+, SqueezeNet 1.1
- Fast training, small models, good for prototyping

**Good Accuracy** (Production quality):
- ResNet-50, EfficientNet B2-B3, ResNeXt-50
- Balanced performance, reliable results

**High Accuracy** (State-of-the-art):
- EfficientNet B4-B7, EfficientNet V2 series, ResNeXt-101
- Best possible results, higher computational cost

## Configuration Best Practices

### Pretrained Weights

**Recommendation**: Always use `"pretrained": true` unless you have specific reasons not to.

**Benefits**:
- Faster convergence during training
- Better final accuracy, especially on smaller datasets  
- More stable training process
- Requires less training data to achieve good results

**When to disable**:
- Training from scratch for research purposes
- Very domain-specific images where ImageNet pretraining may not help
- When you want to understand training dynamics without transfer learning

### Model Selection Process

1. **Start with ResNet-50**: Good baseline for most tasks
2. **Optimize for constraints**: Switch to lighter models if needed
3. **Scale up for accuracy**: Move to EfficientNet or larger ResNet/ResNeXt
4. **Validate on your data**: Some models may work better for specific domains

### Input Size Considerations

Different architectures have optimal input sizes:

**Standard Sizes**:
- **224x224**: Default for most models, good starting point
- **256x256**: Better for detailed images, moderate increase in computation
- **384x384**: High detail capture, significant computational increase

**Architecture-Specific Recommendations**:
- **MobileNet/ShuffleNet**: 224x224 optimal, avoid larger sizes
- **SqueezeNet**: 224x224 recommended, designed for efficiency
- **ResNet/ResNeXt**: 224-384x384 depending on variant size
- **EfficientNet**: 224-512x512 depending on variant (B0: 224, B7: 600)

## Advanced Configuration

### Mixed Precision Training

Automatic mixed precision (AMP) is enabled by default and supported by all architectures:

**Highest Benefit**:
- EfficientNet series (complex operations benefit most)
- ResNeXt variants (grouped convolutions)

**Moderate Benefit**:
- ResNet variants
- Larger models in general

**Lower Benefit** (still recommended):
- Lightweight models (MobileNet, ShuffleNet, SqueezeNet)

### Memory Optimization

The coordinator applies several memory-aware features automatically during training preparation:

- Automatic batch size selection via `system.memory.get_optimal_batch_size`, honoring the configured target usage and safety margin
- Channels-last tensor layout for eligible models when `training.runtime.channels_last` is enabled
- Live memory telemetry and recommendations surfaced through the memory API (`/memory/status`)

## Troubleshooting

### Common Issues

**Out of Memory Errors**:
1. Reduce batch size or let system auto-calculate
2. Choose a smaller model variant
3. Reduce input image size
4. Lower `memory.target_memory_usage` or increase `memory.safety_margin` to force a smaller recommended batch

**Poor Training Results**:
1. Ensure pretrained=true for transfer learning
2. Check learning rate (try lower values)
3. Verify your dataset format and labels
4. Consider a larger model if accuracy is insufficient

**Slow Training**:
1. Use lighter models (MobileNet, ShuffleNet) for prototyping
2. Reduce input size if image detail isn't critical
3. Increase batch size if memory allows
4. Check data loading bottlenecks

**Model Selection Confusion**:
1. Start with ResNet-50 as baseline
2. Use performance characteristics table for guidance
3. Test multiple variants if accuracy requirements are strict
4. Consider deployment constraints early in selection

This comprehensive guide covers all 28 supported model variants across 6 architecture families. Each architecture serves specific use cases, from lightweight mobile deployment to state-of-the-art accuracy requirements. The key is matching your specific needs for accuracy, speed, memory usage, and deployment environment to the appropriate model characteristics.

_Page created by Roxxy (AI) – 2025-10-01._