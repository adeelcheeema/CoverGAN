# CoverGAN: Cover Photo Generation from Text Story Using Layout-Guided GAN

[![Paper DOI](https://img.shields.io/badge/Paper-Springer-blue)](https://link.springer.com/article/10.1007/s00500-025-10436-y)

This repository contains the implementation of **CoverGAN**, a deep learning model that generates cover photos from textual stories using a layout-guided Generative Adversarial Network (GAN). The work has been published in **Soft Computing (Springer, 2025)**.

## 📖 Research Paper
This work has been published in *Soft Computing (Springer)*:
🔗 **[Read the Paper](https://link.springer.com/article/10.1007/s00500-025-10436-y)**


## 🚀 Features
- **Text-to-Image Generation**: Converts textual descriptions into structured cover images.
- **Layout-Guided GAN**: Uses predefined layouts to control object placement.
- **Diffusion-Based Enhancements**: Implements diffusion models for high-quality image generation.
- **Evaluation Metrics**: Includes Inception Score (IS) and Fréchet Inception Distance (FID) for quality assessment.

## 🛠 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/adeelcheeema/CoverGAN.git
cd CoverGAN
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 4️⃣ Generate Images
To generate images from a story text:
```bash
python test.py --input stories/sample_story.txt --output output.png
```

## 📜 Citation
If you use this code in your research, please cite our paper:

```
@article{cheeema2025covergan,
  author = {Adeel Cheeema et al.},
  title = {CoverGAN: Cover Photo Generation from Text Story Using Layout-Guided GAN},
  journal = {Soft Computing},
  year = {2025},
  publisher = {Springer}
}
```

## Contact
For any questions or collaborations, feel free to reach out!
📧 Email: adeelcheema2011@gmail.com
🐦 Twitter: [@Adeelcheema2011](https://x.com/Adeelcheema2011)

