# FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction

<p align="center">
  <em>AAAI 2026</em>
</p>

> **FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction**
>
> Haowei Zhang<sup>1,2</sup>, Yuanpei Zhao<sup>1,2</sup>, Jizhe Zhou<sup>1,2\*</sup>, Mao Li<sup>1,2\*</sup>
>
> <sup>1</sup> College of Computer Science, Sichuan University, China
> <sup>2</sup> Engineering Research Center of Machine Learning and Industry Intelligence, Ministry of Education of China
>
> <sup>\*</sup> Corresponding authors

## Abstract

Improving the diversity of generated results while maintaining high visual quality remains a significant challenge in image generation tasks. Fractal Generative Models (FGMs) are efficient in generating high-quality images, but their inherent self-similarity limits the diversity of output images. To address this issue, we propose a novel approach based on the Hausdorff Dimension (HD), a widely recognized concept in fractal geometry used to quantify structural complexity, which aids in enhancing the diversity of generated outputs. To incorporate HD into FGM, we propose a learnable HD estimation method that predicts HD directly from image embeddings. During training, we adopt an HD-based loss with a Monotonic Momentum-Driven Scheduling (MMDS) strategy to progressively obtain optimal diversity without sacrificing visual quality. During inference, we employ HD-guided rejection sampling to select geometrically richer outputs. Extensive experiments on ImageNet demonstrate that our FGM-HD framework yields a **39% improvement in output diversity** compared to vanilla FGMs, while preserving comparable image quality.

## Key Contributions

- **Introducing Hausdorff Dimension for Diversity Enhancement**: The first work to introduce HD into the FGM framework, alleviating the self-similarity limitation inherent in fractal-based generation.
- **Learnable and Efficient HD Estimation**: A ResNet152-based multi-scale network that predicts HD directly from image embeddings, enabling scalable integration into generative frameworks.
- **Monotonic Momentum-Driven Scheduling (MMDS)** : A dynamic weight adjustment strategy that balances visual quality and structural diversity during training, extendable to other models with hybrid loss functions.
- **HD-Guided Rejection Sampling**: An inference-time strategy that filters outputs by HD threshold, resulting in geometrically richer and more diverse generated images.

## Citation

```bibtex
@inproceedings{zhang2026fgmhd,
  title={FGM-HD: Boosting Generation Diversity of Fractal Generative Models through Hausdorff Dimension Induction},
  author={Zhang, Haowei and Zhao, Yuanpei and Zhou, Jizhe and Li, Mao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgements

This project builds upon [Fractal Generative Models](https://github.com/LTH14/fractalgen) by Li et al. (2025).

## License

MIT License — see [LICENSE](LICENSE).
