# Manga-Style Conversion of Natural Images via Adaptive Screentones

## Setup
```bash
pip install -r requirements.txt
```

## Code Execution
Generate screentone library:
```bash
python generate_screentone.py
```
Convert image to manga style:
```bash
python convert_manga.py
```
You can modify the parameters at the top of the `convert_manga.py`, including input/output file paths, screentone types to use and whether to apply histogram equalization.

## Examples
### Example1
NTU library:
![NTU library picture](pictures/input_example1.jpg)

Converted manga style:
![Manga style picture](pictures/output_example1.png)

### Example2
City street:
![City street picture](pictures/input_example2.png)

Converted manga style:
![Manga style picture](pictures/output_example2.png)