# StabDiff - Stable Diffusion Application

A simple and user-friendly web application for generating images using Stable Diffusion AI model. Create stunning images from text descriptions with an intuitive interface.

## Features

- 🎨 Generate images from text prompts
- 🖼️ Customizable image dimensions (128px to 1024px)
- ⚙️ Adjustable generation parameters (steps, guidance scale)
- 🎲 Reproducible results with seed control
- 🚫 Negative prompts to avoid unwanted elements
- 💻 Web-based interface using Gradio
- 🔥 GPU acceleration support (CUDA)

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended, but CPU mode is supported)
- At least 8GB RAM (16GB+ recommended for GPU)
- ~5GB disk space for model download

## Installation

1. Clone this repository:
```bash
git clone https://github.com/pkmariya/StabDiff.git
cd StabDiff
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

3. Enter your text prompt and adjust parameters as needed

4. Click "Submit" to generate your image

## Parameters

- **Prompt**: Description of the image you want to generate
- **Negative Prompt**: Elements you want to avoid in the image (optional)
- **Number of Steps**: More steps = higher quality but slower (1-100, default: 50)
- **Guidance Scale**: How closely to follow the prompt (1-20, default: 7.5)
- **Width/Height**: Output image dimensions (128-1024, default: 512x512)
- **Seed**: Use -1 for random, or a specific number for reproducible results

## Example Prompts

- "a beautiful sunset over mountains"
- "a cute cat playing with a ball of yarn"
- "a futuristic city at night with neon lights"
- "a serene lake surrounded by autumn trees"
- "an astronaut riding a horse on mars"

## Configuration

You can modify `config.json` to change default settings:
- Model selection
- Default generation parameters
- Server settings

## Troubleshooting

### Out of Memory Error
- Reduce image dimensions to 256x256 or 384x384
- Use fewer inference steps
- Restart the application

### Slow Generation
- Enable GPU if available
- Reduce number of inference steps
- Use smaller image dimensions

### Model Download Issues
- Ensure you have stable internet connection
- Check if you have enough disk space (~5GB)
- Models are cached in `~/.cache/huggingface/`

## License

This project is open source and available for educational purposes.

## Credits

- Built with [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- UI powered by [Gradio](https://gradio.app/)
- Uses Stable Diffusion model from [RunwayML](https://huggingface.co/runwayml/stable-diffusion-v1-5)

## Disclaimer

Please use this application responsibly. Generated images should comply with ethical guidelines and local laws.